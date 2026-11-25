"""大视频分片上传：前端 JS 切片 + 后端 FastAPI 端点合并落盘。

Gradio 的 gr.File 是整块上传（受内存/大小限制），几 GB 的视频会失败。
这里绕开 gr.File：前端用 File.slice() 分片 + fetch 并发 POST 到自定义端点，
后端把分片写入临时目录、全部到齐后按序合并到输入目录，再触发前端刷新列表。
"""
import re
import shutil
from pathlib import Path

from fastapi import File, Form, HTTPException, UploadFile

import status as st

REPO_ROOT = Path(__file__).resolve().parents[2]
UPLOAD_TMP_DIR = REPO_ROOT / "logs" / "webui" / "upload_tmp"

_UPLOAD_ID_RE = re.compile(r"^[A-Za-z0-9_-]{8,64}$")
_MAX_CHUNK_BYTES = 32 * 1024 * 1024   # 单片上限 32MB
_MAX_TOTAL_CHUNKS = 20000             # 8MB * 20000 ≈ 156GB，足够大


def _safe_upload_id(upload_id: str) -> str:
    if not upload_id or not _UPLOAD_ID_RE.match(upload_id):
        raise HTTPException(status_code=400, detail="非法 upload_id")
    return upload_id


def _safe_name(filename: str) -> str:
    name = Path(filename or "").name.strip()
    if not name:
        raise HTTPException(status_code=400, detail="文件名不能为空")
    if Path(name).suffix.lower() not in st.EP_EXTS:
        raise HTTPException(status_code=400, detail="不支持的文件类型")
    return name


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


def register_routes(app):
    @app.post("/upload/chunk")
    def upload_chunk(
        upload_id: str = Form(...),
        filename: str = Form(...),
        chunk_index: int = Form(...),
        total_chunks: int = Form(...),
        chunk: UploadFile = File(...),
    ):
        upload_id = _safe_upload_id(upload_id)
        name = _safe_name(filename)
        if not 1 <= total_chunks <= _MAX_TOTAL_CHUNKS:
            raise HTTPException(status_code=400, detail="分片总数非法")
        if not 0 <= chunk_index < total_chunks:
            raise HTTPException(status_code=400, detail="分片序号非法")
        data = chunk.file.read()
        if len(data) > _MAX_CHUNK_BYTES:
            raise HTTPException(status_code=400, detail="单片过大")
        d = _chunk_dir(upload_id)
        (d / ".meta").write_text(name, encoding="utf-8")
        (d / f"chunk_{chunk_index}").write_bytes(data)
        return {"ok": True, "received": chunk_index}

    @app.post("/upload/complete")
    def upload_complete(
        upload_id: str = Form(...),
        filename: str = Form(...),
        total_chunks: int = Form(...),
    ):
        upload_id = _safe_upload_id(upload_id)
        name = _safe_name(filename)
        if not 1 <= total_chunks <= _MAX_TOTAL_CHUNKS:
            raise HTTPException(status_code=400, detail="分片总数非法")
        d = UPLOAD_TMP_DIR / upload_id
        if not d.exists():
            raise HTTPException(status_code=404, detail="未找到上传会话")
        missing = [i for i in range(total_chunks) if not (d / f"chunk_{i}").exists()]
        if missing:
            raise HTTPException(status_code=400, detail=f"缺少分片: {missing[:5]}")
        dest_dir = st.input_dir()
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = _unique_dest(dest_dir / name)
        with open(dest, "wb") as out:
            for i in range(total_chunks):
                with open(d / f"chunk_{i}", "rb") as f:
                    shutil.copyfileobj(f, out)
        shutil.rmtree(d, ignore_errors=True)
        return {"ok": True, "name": dest.name}


# ── 前端 JS（注入 Blocks(js=...)）────────────────────────────────

UPLOAD_JS = """
(() => {
  const CHUNK_SIZE = 8 * 1024 * 1024;   // 8MB/片
  const CONCURRENCY = 3;                 // 并发分片数
  const STYLE = `
    .chunk-upload { margin: 4px 0 18px; }
    .cu-drop { border: 2px dashed #3a4356; border-radius: 12px; padding: 30px 16px;
      text-align: center; cursor: pointer; background: #141924;
      transition: border-color .2s, background .2s; }
    .cu-drop:hover, .cu-drop.dragging { border-color: #f97316; background: #181f2c; }
    .cu-drop-icon { font-size: 30px; opacity: .75; }
    .cu-drop-title { margin-top: 10px; font-size: 15px; font-weight: 600; color: #e6e8ec; }
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
        <div class="cu-drop-sub">支持 mp4 / avi / mov / mkv / flv / wmv，大文件自动分片上传</div>
        <input type="file" id="cu-file" multiple
          accept=".mp4,.avi,.mov,.mkv,.flv,.wmv,video/*" hidden />
      </div>
      <div class="cu-list" id="cu-list"></div>`;
    const iframe = anchor.querySelector('iframe');
    if (iframe) iframe.remove();
    anchor.appendChild(root);

    const drop = root.querySelector('#cu-drop');
    const input = root.querySelector('#cu-file');
    drop.addEventListener('click', () => input.click());
    input.addEventListener('change', () => { handleFiles(input.files); input.value = ''; });
    ['dragover', 'dragenter'].forEach(ev =>
      drop.addEventListener(ev, e => { e.preventDefault(); drop.classList.add('dragging'); }));
    ['dragleave', 'drop'].forEach(ev =>
      drop.addEventListener(ev, e => { e.preventDefault(); drop.classList.remove('dragging'); }));
    drop.addEventListener('drop', e => handleFiles(e.dataTransfer.files));
    anchor.dataset.uploadReady = '1';
    return true;
  }

  function createItem(name) {
    const el = document.createElement('div');
    el.className = 'cu-item';
    el.innerHTML = `
      <div class="cu-name"></div>
      <div class="cu-bar"><div class="cu-bar-fill"></div></div>
      <div class="cu-status">排队中</div>`;
    el.querySelector('.cu-name').textContent = name;
    return el;
  }

  async function handleFiles(files) {
    const list = document.getElementById('cu-list');
    for (const file of files) {
      const item = createItem(file.name);
      list.appendChild(item);
      while (uploading) { await new Promise(r => setTimeout(r, 300)); }
      uploading = true;
      try {
        await uploadOne(file, pct => {
          item.querySelector('.cu-bar-fill').style.width = pct + '%';
          item.querySelector('.cu-status').textContent =
            pct < 100 ? ('上传中 ' + pct + '%') : '合并中…';
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
    let next = 0, done = 0;
    const worker = async () => {
      while (true) {
        const idx = next++;
        if (idx >= total) break;
        await sendChunk(uploadId, file, idx, total);
        done++;
        onProgress(Math.round(done / total * 100));
      }
    };
    await Promise.all(Array.from({ length: Math.min(CONCURRENCY, total) }, worker));
    await complete(uploadId, file.name, total);
  }

  async function sendChunk(uploadId, file, idx, total) {
    const start = idx * CHUNK_SIZE;
    const end = Math.min(start + CHUNK_SIZE, file.size);
    const fd = new FormData();
    fd.append('upload_id', uploadId);
    fd.append('filename', file.name);
    fd.append('chunk_index', String(idx));
    fd.append('total_chunks', String(total));
    fd.append('chunk', file.slice(start, end), file.name);
    const resp = await fetch('/upload/chunk', { method: 'POST', body: fd });
    if (!resp.ok) {
      const j = await resp.json().catch(() => ({}));
      throw new Error(j.detail || ('分片上传失败 ' + resp.status));
    }
  }

  async function complete(uploadId, name, total) {
    const fd = new FormData();
    fd.append('upload_id', uploadId);
    fd.append('filename', name);
    fd.append('total_chunks', String(total));
    const resp = await fetch('/upload/complete', { method: 'POST', body: fd });
    if (!resp.ok) {
      const j = await resp.json().catch(() => ({}));
      throw new Error(j.detail || ('合并失败 ' + resp.status));
    }
    const btn = document.getElementById('refresh-videos');
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
