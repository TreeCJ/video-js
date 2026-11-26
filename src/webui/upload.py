"""大视频分片上传：前端 JS 切片 + 后端 FastAPI 端点合并落盘。

Gradio 的 gr.File 是整块上传（受内存/大小限制），几 GB 的视频会失败。
这里绕开 gr.File：前端用 File.slice() 分片 + fetch 并发 POST 到自定义端点，
后端把分片写入临时目录、全部到齐后按序合并到输入目录，再触发前端刷新列表。

协议（multipart/form-data）：
  POST /upload/chunk     upload_id, filename, chunk_index, total_chunks,
                         chunk_size, file_size, chunk=<File>
  POST /upload/complete  upload_id, filename, total_chunks, chunk_size, file_size

设计要点：
  * 前端固定 8MB/片、3 并发、单片失败自动重试 3 次；
  * 每片写入独立文件 chunk_<i>，同会话多片可并发到达；
  * 服务端按 chunk_size/file_size 强校验每片字节数，截断的分片当场 400；
  * complete 先把会话目录原子改名为 .merging-<id>（防重复合并），
    校验缺片/片长 → 磁盘余量 → 流式合并到 <dest>.part → ffprobe 校验 → os.replace 原子落盘；
  * 同名文件自动追加 _1/_2，不覆盖库内已有剧集；
  * 孤儿会话与 .part 残片由 cleanup_stale()（启动时调用）按 24h 超时清理。
"""
import os
import re
import shutil
import subprocess
import time
from pathlib import Path

from fastapi import File, Form, HTTPException, UploadFile

import status as st

REPO_ROOT = Path(__file__).resolve().parents[2]
UPLOAD_TMP_DIR = REPO_ROOT / "logs" / "webui" / "upload_tmp"

_UPLOAD_ID_RE = re.compile(r"^[A-Za-z0-9_-]{8,64}$")
_MAX_CHUNK_BYTES = 32 * 1024 * 1024        # 单片上限 32MB（前端实际用 8MB）
_MAX_TOTAL_CHUNKS = 20000                  # 8MB * 20000 ≈ 156GB，足够大
_MAX_NAME_BYTES = 240                      # 文件名 UTF-8 编码长度上限
_STALE_SECONDS = 24 * 3600                 # 孤儿分片会话保留 24h
_MERGE_MARGIN_BYTES = 128 * 1024 * 1024    # 合并前要求目标盘至少多留 128MB
_COPY_BUFSIZE = 1024 * 1024


# ── 参数校验 ────────────────────────────────────────────────────────

def _safe_upload_id(upload_id: str) -> str:
    if not upload_id or not _UPLOAD_ID_RE.match(upload_id):
        raise HTTPException(status_code=400, detail="非法 upload_id")
    return upload_id


def _safe_name(filename: str) -> str:
    name = Path(filename or "").name.strip()
    if not name:
        raise HTTPException(status_code=400, detail="文件名不能为空")
    if len(name.encode("utf-8")) > _MAX_NAME_BYTES:
        raise HTTPException(status_code=400, detail="文件名过长")
    if Path(name).suffix.lower() not in st.EP_EXTS:
        raise HTTPException(status_code=400, detail="不支持的文件类型")
    return name


def _validate_meta(chunk_size: int, file_size: int, total_chunks: int):
    """校验 chunk_size / file_size / total_chunks 三者自洽，返回 (cs, fs, total)。"""
    if not 1 <= chunk_size <= _MAX_CHUNK_BYTES:
        raise HTTPException(status_code=400, detail="单片大小非法")
    if not 1 <= file_size <= chunk_size * _MAX_TOTAL_CHUNKS:
        raise HTTPException(status_code=400, detail="文件大小非法")
    expected_total = (file_size + chunk_size - 1) // chunk_size
    if total_chunks != expected_total:
        raise HTTPException(status_code=400,
                            detail=f"分片总数与文件大小不符（应为 {expected_total}）")
    return chunk_size, file_size, total_chunks


def _expected_chunk_len(chunk_size: int, file_size: int, idx: int, total: int) -> int:
    """第 idx 片的应有字节数：除最后一片外都应是整片，最后一片是余数。"""
    if idx < total - 1:
        return chunk_size
    return file_size - (total - 1) * chunk_size


def _chunk_dir(upload_id: str) -> Path:
    d = UPLOAD_TMP_DIR / upload_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def _unique_dest(dest: Path) -> Path:
    """同名文件不覆盖，追加 _1/_2 后缀，避免误覆盖已在库里的剧集。"""
    if not dest.exists():
        return dest
    stem, suffix = dest.stem, dest.suffix
    i = 1
    while True:
        cand = dest.with_name(f"{stem}_{i}{suffix}")
        if not cand.exists():
            return cand
        i += 1


def _probe_video(path: Path) -> bool:
    """ffprobe 能读出正时长视为有效视频；机器上没有 ffprobe 则跳过校验。"""
    if shutil.which("ffprobe") is None:
        return True
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
        capture_output=True, text=True)
    if r.returncode != 0:
        return False
    try:
        return float(r.stdout.strip()) > 0
    except ValueError:
        return False


# ── 路由 ────────────────────────────────────────────────────────────

def register_routes(app):
    @app.post("/upload/chunk")
    def upload_chunk(
        upload_id: str = Form(...),
        filename: str = Form(...),
        chunk_index: int = Form(...),
        total_chunks: int = Form(...),
        chunk_size: int = Form(...),
        file_size: int = Form(...),
        chunk: UploadFile = File(...),
    ):
        upload_id = _safe_upload_id(upload_id)
        name = _safe_name(filename)
        cs, fs, total = _validate_meta(chunk_size, file_size, total_chunks)
        if not 0 <= chunk_index < total:
            raise HTTPException(status_code=400, detail="分片序号非法")
        expected = _expected_chunk_len(cs, fs, chunk_index, total)
        data = chunk.file.read()
        # 严格按协议校验字节数：客户端切片 bug / 请求被截断都在这里当场发现，
        # 而不是等全部传完合并时才发现文件损坏。
        if len(data) != expected:
            raise HTTPException(status_code=400,
                                detail=f"分片大小不符：期望 {expected} 字节，收到 {len(data)} 字节")
        d = _chunk_dir(upload_id)
        (d / ".meta").write_text(name, encoding="utf-8")
        (d / f"chunk_{chunk_index}").write_bytes(data)
        return {"ok": True, "received": chunk_index}

    @app.post("/upload/complete")
    def upload_complete(
        upload_id: str = Form(...),
        filename: str = Form(...),
        total_chunks: int = Form(...),
        chunk_size: int = Form(...),
        file_size: int = Form(...),
    ):
        upload_id = _safe_upload_id(upload_id)
        name = _safe_name(filename)
        cs, fs, total = _validate_meta(chunk_size, file_size, total_chunks)

        # 原子锁：会话目录改名 .merging-<id>，并发 complete 只有一个能成功，
        # 另一个拿不到原目录直接 404，不会重复合并。
        d = UPLOAD_TMP_DIR / upload_id
        locked = UPLOAD_TMP_DIR / (".merging-" + upload_id)
        if not d.exists():
            raise HTTPException(status_code=404, detail="未找到上传会话（可能已合并或已过期）")
        try:
            d.rename(locked)
        except FileNotFoundError:
            raise HTTPException(status_code=404, detail="未找到上传会话（可能已合并或已过期）")
        except OSError:
            raise HTTPException(status_code=409, detail="该上传正在合并中，请勿重复提交")

        try:
            try:
                meta_name = (locked / ".meta").read_text(encoding="utf-8").strip()
            except OSError:
                raise HTTPException(status_code=400, detail="会话元数据缺失，请重新上传")
            if meta_name != name:
                raise HTTPException(status_code=400, detail="文件名与会话记录不一致")

            missing = [i for i in range(total) if not (locked / f"chunk_{i}").exists()]
            if missing:
                raise HTTPException(status_code=400,
                                    detail=f"缺少分片: {missing[:5]}{'…' if len(missing) > 5 else ''}")
            for i in range(total):
                expect = _expected_chunk_len(cs, fs, i, total)
                try:
                    got = (locked / f"chunk_{i}").stat().st_size
                except OSError:
                    got = -1
                if got != expect:
                    raise HTTPException(
                        status_code=400,
                        detail=f"分片 {i} 大小不符：期望 {expect} 字节，实际 {got} 字节")

            dest_dir = st.input_dir()
            dest_dir.mkdir(parents=True, exist_ok=True)
            free = shutil.disk_usage(dest_dir).free
            if free < fs + _MERGE_MARGIN_BYTES:
                raise HTTPException(
                    status_code=507,
                    detail=f"磁盘空间不足：需要约 {(fs + _MERGE_MARGIN_BYTES) / 1024 ** 3:.1f} GB，"
                           f"剩余 {free / 1024 ** 3:.1f} GB")

            # 合并到 .part（EP_EXTS 之外，剧集列表不会扫到），成功后原子改名。
            dest = _unique_dest(dest_dir / name)
            part = dest_dir / (dest.name + ".part")
            try:
                with open(part, "wb") as out:
                    for i in range(total):
                        with open(locked / f"chunk_{i}", "rb") as f:
                            shutil.copyfileobj(f, out, length=_COPY_BUFSIZE)
                if part.stat().st_size != fs:
                    raise OSError(f"合并后大小 {part.stat().st_size} != 声明大小 {fs}")
                if not _probe_video(part):
                    raise HTTPException(
                        status_code=422,
                        detail="ffprobe 校验失败：文件不是有效视频（是否传错文件或传输损坏？）")
                # 用硬链接原子落盘（目标已存在则 EEXIST 而非覆盖），
                # 并发上传同名文件时也绝不会互相覆盖。
                while True:
                    try:
                        os.link(part, dest)
                        break
                    except FileExistsError:
                        dest = _unique_dest(dest)
                part.unlink()
            except HTTPException:
                part.unlink(missing_ok=True)
                raise
            except OSError as e:
                part.unlink(missing_ok=True)
                raise HTTPException(status_code=500, detail=f"合并写入失败：{e}")
            shutil.rmtree(locked, ignore_errors=True)
            return {"ok": True, "name": dest.name, "size": fs}
        except BaseException:
            # 失败即清场，避免残留 .merging-* 目录；cleanup_stale 兜底。
            shutil.rmtree(locked, ignore_errors=True)
            raise


def cleanup_stale(max_age_seconds: int = _STALE_SECONDS):
    """清理超时的分片会话目录与输入目录里的 .part 残片（应用启动时调用）。"""
    now = time.time()

    def _stale(p: Path) -> bool:
        try:
            return now - p.stat().st_mtime > max_age_seconds
        except OSError:
            return False

    if UPLOAD_TMP_DIR.exists():
        for d in UPLOAD_TMP_DIR.iterdir():
            if not _stale(d):
                continue
            try:
                if d.is_dir():
                    shutil.rmtree(d, ignore_errors=True)
                else:
                    d.unlink(missing_ok=True)
            except OSError:
                pass
    try:
        in_dir = st.input_dir()
    except OSError:
        return
    if in_dir.exists():
        for p in in_dir.iterdir():
            if p.is_file() and p.name.endswith(".part") and _stale(p):
                try:
                    p.unlink(missing_ok=True)
                except OSError:
                    pass


# ── 前端 JS（注入 Blocks(js=...)）────────────────────────────────

UPLOAD_JS = """
(() => {
  const CHUNK_SIZE = 8 * 1024 * 1024;   // 8MB/片
  const CONCURRENCY = 3;                 // 并发分片数
  const MAX_RETRY = 3;                   // 单片失败重试次数
  const STYLE = `
    .chunk-upload { margin: 2px 0 14px; }
    .cu-drop { border: 2px dashed #3a4356; border-radius: 12px; padding: 22px 16px;
      text-align: center; cursor: pointer; background: #141924;
      transition: border-color .2s, background .2s; }
    .cu-drop:hover, .cu-drop.dragging { border-color: #f97316; background: #181f2c; }
    .cu-drop-icon { font-size: 26px; opacity: .75; }
    .cu-drop-title { margin-top: 8px; font-size: 15px; font-weight: 600; color: #e6e8ec; }
    .cu-drop-sub { margin-top: 4px; font-size: 12px; color: #8a93a6; }
    .cu-list { display: flex; flex-direction: column; gap: 8px; margin-top: 10px; }
    .cu-item { background: #141924; border: 1px solid #232a36; border-radius: 8px; padding: 10px 12px; }
    .cu-name { font-size: 13px; color: #e6e8ec; word-break: break-all; }
    .cu-bar { height: 6px; background: #232a36; border-radius: 3px; margin-top: 8px; overflow: hidden; }
    .cu-bar-fill { height: 100%; width: 0; background: linear-gradient(90deg, #f97316, #fb923c);
      border-radius: 3px; transition: width .2s; }
    .cu-status { margin-top: 6px; font-size: 12px; color: #8a93a6; }
    .cu-item.done .cu-status { color: #4ade80; }
    .cu-item.fail .cu-status { color: #f87171; }
  `;

  function uid() {
    if (window.crypto && crypto.randomUUID) return crypto.randomUUID();
    return 'u' + Date.now().toString(36) + Math.random().toString(36).slice(2, 10);
  }

  function fmtMB(b) { return (b / 1048576).toFixed(0) + ' MB'; }
  function fmtSpeed(bps) {
    if (bps >= 1048576) return (bps / 1048576).toFixed(1) + ' MB/s';
    return (bps / 1024).toFixed(0) + ' KB/s';
  }

  function injectStyle() {
    if (document.getElementById('chunk-upload-style')) return;
    const s = document.createElement('style');
    s.id = 'chunk-upload-style';
    s.textContent = STYLE;
    document.head.appendChild(s);
  }

  let uploading = false;

  function buildUI() {
    const anchor = document.getElementById('upload-anchor');
    if (!anchor) return false;
    if (anchor.dataset.uploadReady === '1') return true;
    injectStyle();
    const root = document.createElement('div');
    root.className = 'chunk-upload';
    root.innerHTML = `
      <div class="cu-drop" id="cu-drop">
        <div class="cu-drop-icon">&#11014;</div>
        <div class="cu-drop-title">拖拽视频到这里，或点击选择文件</div>
        <div class="cu-drop-sub">支持 mp4 / avi / mov / mkv / flv / wmv · 8MB/片自动分片 · 单片失败自动重试 · 完成后自动刷新剧集列表</div>
        <input type="file" id="cu-file" multiple
          accept=".mp4,.avi,.mov,.mkv,.flv,.wmv,video/*" hidden />
      </div>
      <div class="cu-list" id="cu-list"></div>`;
    // gr.HTML 在部分 Gradio 版本会把内容渲染进 iframe，一并移除
    const iframe = anchor.querySelector('iframe');
    if (iframe) iframe.remove();
    anchor.appendChild(root);

    const drop = root.querySelector('#cu-drop');
    const input = root.querySelector('#cu-file');
    drop.addEventListener('click', () => input.click());
    input.addEventListener('change', () => { handleFiles(input.files); input.value = ''; });
    ['dragover', 'dragenter'].forEach(ev =>
      drop.addEventListener(ev, e => { e.preventDefault(); e.stopPropagation(); drop.classList.add('dragging'); }));
    ['dragleave', 'drop'].forEach(ev =>
      drop.addEventListener(ev, e => { e.preventDefault(); e.stopPropagation(); drop.classList.remove('dragging'); }));
    drop.addEventListener('drop', e => handleFiles(e.dataTransfer.files));
    // 防止文件拖到页面其他位置时浏览器直接打开该文件
    ['dragover', 'drop'].forEach(ev => window.addEventListener(ev, e => e.preventDefault()));
    anchor.dataset.uploadReady = '1';
    return true;
  }

  function createItem(name, size) {
    const el = document.createElement('div');
    el.className = 'cu-item';
    el.innerHTML = `
      <div class="cu-name"></div>
      <div class="cu-bar"><div class="cu-bar-fill"></div></div>
      <div class="cu-status">排队中 · ${fmtMB(size)}</div>`;
    el.querySelector('.cu-name').textContent = name;
    return el;
  }

  async function handleFiles(files) {
    const list = document.getElementById('cu-list');
    for (const file of files) {
      const item = createItem(file.name, file.size);
      list.appendChild(item);
      while (uploading) { await new Promise(r => setTimeout(r, 300)); }
      uploading = true;
      item.querySelector('.cu-status').textContent = '上传中 0% · ' + fmtMB(file.size);
      try {
        await uploadOne(file, (pct, bytes, speed) => {
          item.querySelector('.cu-bar-fill').style.width = pct + '%';
          item.querySelector('.cu-status').textContent = pct < 100
            ? `上传中 ${pct}% · ${fmtMB(bytes)} / ${fmtMB(file.size)}`
              + (speed ? ` · ${fmtSpeed(speed)}` : '')
            : '合并校验中…';
        });
        item.querySelector('.cu-status').textContent = '完成';
        item.classList.add('done');
      } catch (err) {
        item.querySelector('.cu-status').textContent = '失败：' + (err && err.message ? err.message : err);
        item.classList.add('fail');
      } finally {
        uploading = false;
      }
    }
  }

  async function uploadOne(file, onProgress) {
    const total = Math.max(1, Math.ceil(file.size / CHUNK_SIZE));
    const uploadId = uid();
    const t0 = Date.now();
    let next = 0, done = 0;
    const worker = async () => {
      while (true) {
        const idx = next++;
        if (idx >= total) break;
        await sendChunk(uploadId, file, idx, total);
        done++;
        const pct = Math.round(done / total * 100);
        const bytes = Math.min(done * CHUNK_SIZE, file.size);
        const elapsed = (Date.now() - t0) / 1000;
        onProgress(pct, bytes, elapsed > 0.5 ? bytes / elapsed : 0);
      }
    };
    await Promise.all(Array.from({ length: Math.min(CONCURRENCY, total) }, worker));
    await complete(uploadId, file, total);
  }

  async function sendChunk(uploadId, file, idx, total) {
    const start = idx * CHUNK_SIZE;
    const end = Math.min(start + CHUNK_SIZE, file.size);
    const blob = file.slice(start, end);
    let lastErr;
    for (let attempt = 1; attempt <= MAX_RETRY; attempt++) {
      const fd = new FormData();
      fd.append('upload_id', uploadId);
      fd.append('filename', file.name);
      fd.append('chunk_index', String(idx));
      fd.append('total_chunks', String(total));
      fd.append('chunk_size', String(CHUNK_SIZE));
      fd.append('file_size', String(file.size));
      fd.append('chunk', blob, file.name);
      try {
        const resp = await fetch('/upload/chunk', { method: 'POST', body: fd });
        if (resp.ok) return;
        lastErr = new Error((await resp.json().catch(() => ({}))).detail || ('分片上传失败 ' + resp.status));
      } catch (err) {
        lastErr = err;
      }
      if (attempt < MAX_RETRY) await new Promise(r => setTimeout(r, 500 * attempt));
    }
    throw lastErr;
  }

  async function complete(uploadId, file, total) {
    const fd = new FormData();
    fd.append('upload_id', uploadId);
    fd.append('filename', file.name);
    fd.append('total_chunks', String(total));
    fd.append('chunk_size', String(CHUNK_SIZE));
    fd.append('file_size', String(file.size));
    const resp = await fetch('/upload/complete', { method: 'POST', body: fd });
    if (!resp.ok) {
      throw new Error((await resp.json().catch(() => ({}))).detail || ('合并失败 ' + resp.status));
    }
    refreshVideoList();
  }

  function refreshVideoList() {
    // Gradio 的 elem_id 挂在组件外层 div 上，真实 <button> 在其内部
    const el = document.getElementById('refresh-videos');
    if (!el) return;
    const btn = el.tagName === 'BUTTON' ? el : el.querySelector('button');
    if (btn) btn.click();
  }

  // Gradio 是 Svelte 异步挂载，锚点可能稍后才出现，轮询等待。
  function waitAndBuild() {
    let tries = 0;
    const t = setInterval(() => {
      if (buildUI() || ++tries > 100) clearInterval(t);
    }, 200);
  }
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', waitAndBuild);
  } else {
    waitAndBuild();
  }
})();
"""
