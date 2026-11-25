"""电影解说工作台入口：python src/webui/app.py

代理处理：管线子进程内 LLM/edge-tts 需走代理，但 Gradio 自检 localhost:6006
不能走代理（否则 503）。必须在 import gradio 之前清掉代理；子进程运行时由
runner 把代理装回子进程 env。
"""
import os
import signal
import subprocess
import sys
import time
import copy
import re
import urllib.parse
from pathlib import Path

# 让 from config/pipeline import 生效（脚本目录 src/webui 自动在 path，再补 src）
_SRC = Path(__file__).resolve().parent.parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

# ── 代理剥离（必须在 import gradio 之前）──
_PROXY_KEYS = ("http_proxy", "https_proxy", "all_proxy",
               "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY")
_SAVED_PROXY = {k: os.environ.get(k) for k in _PROXY_KEYS}
for _k in _PROXY_KEYS:
    os.environ.pop(_k, None)

# Gradio 默认临时目录是 /tmp/gradio，多用户机器上可能被别的用户占用（无写权限），
# 指到本用户目录避免 PermissionError（必须在 import gradio 前设置）。
_GRADIO_TMP = Path.home() / ".cache" / "gradio"
_GRADIO_TMP.mkdir(parents=True, exist_ok=True)
os.environ["GRADIO_TEMP_DIR"] = str(_GRADIO_TMP)

import yaml
import gradio as gr

from pipeline import STEPS, timer
from runner import JobRunner, list_history_logs, read_history_log
import status as st
import schema as sc
import background as bg

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_FILE = REPO_ROOT / "config.yaml"

# 步骤中文标签 ↔ key（force/skip 勾选框用）
STEP_LABELS = [(zh, key) for key, _l, zh, can_f, can_s in STEPS if can_f or can_s]
LABEL_TO_KEY = {zh: key for zh, key in STEP_LABELS}


# ── 主题 / 全局样式 / 品牌条 ───────────────────────────────────────

def _build_theme():
    return gr.themes.Base(
        primary_hue=gr.themes.colors.orange,
        secondary_hue=gr.themes.colors.orange,
        neutral_hue=gr.themes.colors.slate,
        font=["Inter", "ui-sans-serif", "system-ui", "-apple-system",
              "Segoe UI", "PingFang SC", "Microsoft YaHei", "sans-serif"],
    ).set(
        body_background_fill="#0f1115",
        body_background_fill_dark="#0f1115",
        body_text_color="#c9ced8",
        body_text_color_dark="#c9ced8",
        body_text_color_subdued="#8a93a6",
        body_text_color_subdued_dark="#8a93a6",
        block_background_fill="#161922",
        block_background_fill_dark="#161922",
        block_border_color="#262c39",
        block_border_color_dark="#262c39",
        block_border_width="1px",
        block_radius="12px",
        block_title_text_color="#e6e8ec",
        block_title_text_color_dark="#e6e8ec",
        block_label_background_fill="#1b2029",
        block_label_background_fill_dark="#1b2029",
        block_label_text_color="#aab3c0",
        block_label_text_color_dark="#aab3c0",
        input_background_fill="#12151d",
        input_background_fill_dark="#12151d",
        input_border_color="#2a3140",
        input_border_color_dark="#2a3140",
        code_background_fill="#12151d",
        code_background_fill_dark="#12151d",
        button_primary_background_fill="#f97316",
        button_primary_background_fill_hover="#fb923c",
        button_primary_text_color="#ffffff",
        button_secondary_background_fill="#232a36",
        button_secondary_background_fill_dark="#232a36",
        button_secondary_background_fill_hover="#2c3543",
        button_secondary_background_fill_hover_dark="#2c3543",
        button_secondary_text_color="#e6e8ec",
        button_secondary_text_color_dark="#e6e8ec",
    )


CSS = """
footer { display: none !important; }
.app-header {
  display: flex; align-items: center; justify-content: space-between;
  padding: 14px 20px; margin-bottom: 10px;
  background: linear-gradient(180deg, #1b2029, #161922);
  border: 1px solid #262c39; border-radius: 12px;
}
.app-brand { display: flex; align-items: center; gap: 12px; }
.app-logo {
  width: 40px; height: 40px; border-radius: 10px; flex: 0 0 auto;
  display: flex; align-items: center; justify-content: center;
  font-size: 18px; color: #fff;
  background: linear-gradient(135deg, #f97316, #ea580c);
  box-shadow: 0 4px 12px rgba(249,115,22,.35);
}
.app-title { font-size: 17px; font-weight: 700; color: #eef0f4; letter-spacing: .3px; }
.app-sub { font-size: 12px; color: #8a93a6; margin-top: 2px; }
.app-meta { display: flex; align-items: center; gap: 6px; font-size: 12px; color: #8a93a6; }
.app-dot { width: 8px; height: 8px; border-radius: 50%; background: #4ade80;
  box-shadow: 0 0 0 3px rgba(74,222,128,.18); }
.card {
  border: 1px solid #262c39 !important; border-radius: 12px !important;
  background: #161922 !important; box-shadow: 0 1px 2px rgba(0,0,0,.25);
  padding: 6px 10px !important;
}
.card-title { margin: 2px 0 10px !important; }
.card-title h3, .card-title h2 { font-size: 15px !important; font-weight: 700 !important;
  color: #eef0f4 !important; margin: 0 !important; }
.actionbar {
  border: 1px solid #262c39 !important; border-radius: 12px !important;
  background: #12151d !important; padding: 10px 12px !important;
}
.pp-block { padding: 4px 2px; }
.pp-head { display: flex; align-items: center; justify-content: space-between; margin-bottom: 10px; }
.pp-status { font-size: 14px; color: #e6e8ec; display: flex; align-items: center; gap: 8px; }
.pp-ep { display: inline-block; padding: 1px 8px; border-radius: 6px;
  background: #2a3140; color: #cbd2dd; font-size: 12px; font-weight: 600; }
.pp-elapsed { font-size: 12px; color: #8a93a6; }
.pp-dot { width: 9px; height: 9px; border-radius: 50%; display: inline-block; flex: 0 0 auto; }
.pp-dot.running { background: #f97316; box-shadow: 0 0 0 3px rgba(249,115,22,.2); }
.pp-dot.ok { background: #4ade80; }
.pp-dot.failed { background: #f87171; }
.pp-dot.pending { background: #3a4356; }
.pp-steps { display: flex; flex-wrap: wrap; gap: 8px; }
.pp-chip { display: inline-flex; align-items: center; gap: 6px; padding: 5px 10px;
  border-radius: 8px; font-size: 12px; border: 1px solid #262c39;
  background: #1b2029; color: #8a93a6; }
.pp-chip .ic { font-weight: 700; }
.pp-chip.ok { color: #d1fae5; border-color: rgba(74,222,128,.35); background: rgba(74,222,128,.10); }
.pp-chip.running { color: #fed7aa; border-color: rgba(249,115,22,.45); background: rgba(249,115,22,.12); }
.pp-chip.cached { color: #bfdbfe; border-color: rgba(96,165,250,.35); background: rgba(96,165,250,.10); }
.pp-chip.warn { color: #fef3c7; border-color: rgba(251,191,36,.4); background: rgba(251,191,36,.10); }
.pp-chip.failed { color: #fecaca; border-color: rgba(248,113,113,.4); background: rgba(248,113,113,.12); }
/* ── 第 2 层：CSS 变量级硬覆盖（亮/暗两种渲染路径都钉死为深色） ── */
:root,
.gradio-container,
.gradio-container.dark,
html.dark,
body.dark {
  color-scheme: dark !important;
  --body-background-fill: #0f1115 !important;
  --body-text-color: #e6e8ec !important;
  --body-text-color-subdued: #8a93a6 !important;
  --background-fill-primary: #161922 !important;
  --background-fill-secondary: #12151d !important;
  --block-background-fill: #161922 !important;
  --block-border-color: #262c39 !important;
  --block-info-text-color: #8a93a6 !important;
  --block-title-text-color: #e6e8ec !important;
  --block-label-background-fill: #1b2029 !important;
  --block-label-text-color: #aab3c0 !important;
  --panel-background-fill: #161922 !important;
  --input-background-fill: #12151d !important;
  --input-background-fill-focus: #161a24 !important;
  --input-border-color: #2a3140 !important;
  --input-placeholder-color: #6b7484 !important;
  --code-background-fill: #12151d !important;
  --border-color-primary: #262c39 !important;
  --color-accent: #f97316 !important;
  --link-text-color: #93c5fd !important;
  --checkbox-label-text-color: #e6e8ec !important;
  --table-text-color: #e6e8ec !important;
  --table-odd-background-fill: #12151d !important;
  --table-even-background-fill: #161922 !important;
  --table-border-color: #262c39 !important;
  /* 中性色梯度的浅色端整体压深，消灭所有白色来源（深色端保持原样） */
  --neutral-50: #1b2029 !important;
  --neutral-100: #1b2029 !important;
  --neutral-200: #232a36 !important;
  --neutral-300: #2c3444 !important;
  --neutral-400: #4a5263 !important;
}
/* 系统深色模式下同样重写，防止 Gradio 在媒体查询里把变量改回亮色 */
@media (prefers-color-scheme: dark) {
  :root, .gradio-container, .gradio-container.dark, html.dark, body.dark {
    color-scheme: dark !important;
    --input-background-fill: #12151d !important;
    --input-background-fill-focus: #161a24 !important;
    --body-text-color: #e6e8ec !important;
    --code-background-fill: #12151d !important;
    --block-background-fill: #161922 !important;
    --table-text-color: #e6e8ec !important;
    --table-odd-background-fill: #12151d !important;
    --table-even-background-fill: #161922 !important;
    --neutral-50: #1b2029 !important;
    --neutral-100: #1b2029 !important;
    --neutral-200: #232a36 !important;
    --neutral-300: #2c3444 !important;
  }
}
/* ── 第 3 层：组件级硬覆盖（高优先级，压过 Gradio 自带样式） ── */
body .gradio-container textarea,
body .gradio-container input[type="text"],
body .gradio-container input[type="number"],
body .gradio-container input[type="password"],
body .gradio-container input[type="email"],
body .gradio-container input[type="search"],
body .gradio-container input[type="url"],
body .gradio-container select {
  background-color: #12151d !important;
  color: #e6e8ec !important;
  border-color: #2a3140 !important;
  -webkit-text-fill-color: #e6e8ec !important;
  caret-color: #e6e8ec !important;
}
body .gradio-container textarea::placeholder,
body .gradio-container input::placeholder { color: #6b7484 !important; }
/* 下拉弹层（选项列表默认白底，常见漏网点） */
body .gradio-container [role="listbox"],
body .gradio-container ul[role="listbox"] {
  background: #161a24 !important;
  border-color: #2a3140 !important;
}
body .gradio-container [role="option"] { color: #e6e8ec !important; }
body .gradio-container [role="option"]:hover,
body .gradio-container [role="option"][aria-selected="true"] { background: #232a36 !important; }
/* 代码编辑器（gr.Code / CodeMirror）：容器 + 语法高亮暗色配色 */
body .gradio-container .cm-editor,
body .gradio-container .cm-editor .cm-scroller,
body .gradio-container .cm-editor .cm-content {
  background-color: #12151d !important;
  color: #e6e8ec !important;
  caret-color: #e6e8ec !important;
}
body .gradio-container .cm-gutters {
  background-color: #1b2029 !important; color: #6b7484 !important;
  border-right: 1px solid #262c39 !important;
}
body .gradio-container .cm-line { color: #e6e8ec !important; }
body .gradio-container .cm-activeLine { background-color: rgba(255,255,255,.04) !important; }
body .gradio-container .tok-keyword, body .gradio-container .cm-keyword { color: #c678dd !important; }
body .gradio-container .tok-string, body .gradio-container .cm-string,
body .gradio-container .cm-string2 { color: #98c379 !important; }
body .gradio-container .tok-number, body .gradio-container .cm-number { color: #d19a66 !important; }
body .gradio-container .tok-comment, body .gradio-container .cm-comment { color: #5c6370 !important; }
body .gradio-container .tok-propertyName, body .gradio-container .cm-property,
body .gradio-container .tok-definition, body .gradio-container .cm-def { color: #61afef !important; }
body .gradio-container .tok-atom, body .gradio-container .cm-atom { color: #56b6c2 !important; }
body .gradio-container .tok-operator, body .gradio-container .cm-operator { color: #abb2bf !important; }
body .gradio-container .tok-variableName, body .gradio-container .cm-variableName,
body .gradio-container .cm-variable { color: #e06c75 !important; }
body .gradio-container .tok-typeName, body .gradio-container .cm-typeName { color: #e5c07b !important; }
body .gradio-container .tok-meta, body .gradio-container .cm-meta { color: #8a93a6 !important; }
body .gradio-container .tok-tagName, body .gradio-container .cm-tag { color: #e06c75 !important; }
body .gradio-container .tok-attributeName, body .gradio-container .cm-attribute { color: #d19a66 !important; }
body .gradio-container .tok-heading, body .gradio-container .cm-heading { color: #61afef !important; }
body .gradio-container .tok-link, body .gradio-container .cm-link { color: #61afef !important; }
"""

HEADER_HTML = """
<div class="app-header">
  <div class="app-brand">
    <span class="app-logo">▶</span>
    <div>
      <div class="app-title">电影解说工作台</div>
      <div class="app-sub">剧集转写 · 剧情分析 · 自动剪辑 · 字幕渲染</div>
    </div>
  </div>
  <div class="app-meta"><span class="app-dot"></span> 服务运行中</div>
</div>
"""


# ── 步骤进度（解析 pipeline 的 [i/N Label] STATUS 整行日志）─────────

_STEP_ICON = {
    "pending": ("◌", "pending"),
    "running": ("▶", "running"),
    "ok": ("✓", "ok"),
    "cached": ("↻", "cached"),
    "skipped": ("⤼", "skipped"),
    "disabled": ("·", "disabled"),
    "warn": ("⚠", "warn"),
    "failed": ("✕", "failed"),
}
_STEP_TXT = {
    "pending": "待处理", "running": "进行中", "ok": "完成", "cached": "缓存",
    "skipped": "跳过", "disabled": "停用", "warn": "警告", "failed": "失败",
}


def _step_states(log: str):
    """解析日志，返回 {step_key: (state, detail)}，按 STEPS 顺序。"""
    states = {key: ("pending", "") for key, _l, _zh, _cf, _cs in STEPS}
    label_to_key = {label: key for key, label, _zh, _cf, _cs in STEPS}
    for line in (log or "").splitlines():
        m = re.match(r"^\s*\[(\d+)/(\d+)\s+([^\]]+)\]\s*(.*)$", line)
        if not m:
            continue
        label = m.group(3).strip()
        rest = m.group(4).strip()
        key = next((k for l, k in label_to_key.items()
                    if label == l or label.startswith(l)), None)
        if key is None:
            continue
        if rest.startswith("START"):
            states[key] = ("running", "")
        elif rest.startswith("OK"):
            states[key] = ("ok", rest[2:].strip())
        elif rest.startswith("CACHED"):
            states[key] = ("cached", rest[6:].strip())
        elif rest.startswith("SKIPPED"):
            states[key] = ("skipped", "")
        elif rest.startswith("DISABLED"):
            states[key] = ("disabled", "")
        elif rest.startswith("WARN"):
            states[key] = ("warn", rest[4:].strip())
        elif rest.startswith("FAILED"):
            states[key] = ("failed", rest[6:].strip())
    return states


def _render_progress(log: str, video_name: str, started_at: float) -> str:
    """渲染「状态行 + 9 步进度条」的 HTML。"""
    states = _step_states(log)
    order = [(key, label, zh) for key, label, zh, _cf, _cs in STEPS]
    running_zh = next((zh for key, _l, zh in order if states[key][0] == "running"), None)
    done = sum(1 for key, _l, _zh in order if states[key][0] in ("ok", "cached"))
    failed = sum(1 for key, _l, _zh in order if states[key][0] == "failed")
    n = len(order)

    if failed:
        head = '<span class="pp-dot failed"></span>失败'
        if running_zh:
            head += f' · {running_zh}'
    elif running_zh:
        head = f'<span class="pp-dot running"></span>{running_zh} 进行中'
    elif done == n:
        head = '<span class="pp-dot ok"></span>全部完成'
    elif done:
        head = f'<span class="pp-dot ok"></span>就绪（{done}/{n} 步有缓存）'
    else:
        head = '<span class="pp-dot pending"></span>等待开始'

    if video_name:
        head = f'<span class="pp-ep">{video_name}</span>' + head
    elapsed = timer(started_at) if started_at else "—"

    chips = []
    for key, _l, zh in order:
        state, detail = states[key]
        icon, cls = _STEP_ICON[state]
        title = f"{zh} · {_STEP_TXT[state]}" + (f" · {detail}" if detail else "")
        chips.append(f'<span class="pp-chip {cls}" title="{title}">'
                     f'<span class="ic">{icon}</span>{zh}</span>')
    return ('<div class="pp-block">'
            f'<div class="pp-head"><span class="pp-status">{head}</span>'
            f'<span class="pp-elapsed">已用 {elapsed}</span></div>'
            f'<div class="pp-steps">{"".join(chips)}</div></div>')


def _raw_config() -> dict:
    try:
        with open(CONFIG_FILE, encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except OSError:
        return {}
def _start_face_job():
    """启动人脸候选子进程（不等待）。stdout 直写日志文件——不能接 PIPE 后不读，
    输出超过管道缓冲(64KB)会和子进程互等死锁。返回 (proc, log_path, timeout_s)。"""
    faces_cfg = _raw_config().get("faces", {}) or {}
    try:
        timeout_seconds = max(30, int(faces_cfg.get("timeout_seconds", 900) or 900))
    except (TypeError, ValueError):
        timeout_seconds = 900
    log_dir = REPO_ROOT / "logs" / "webui"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{time.strftime('%Y%m%d_%H%M%S')}_faces.log"
    cmd = [sys.executable, "-u", str(_SRC / "faces.py"), "--config", str(CONFIG_FILE)]
    lf = open(log_path, "w", encoding="utf-8")
    try:
        proc = subprocess.Popen(cmd, cwd=str(REPO_ROOT), stdout=lf,
                                stderr=subprocess.STDOUT, start_new_session=True)
    finally:
        lf.close()   # 子进程已持有自己的 fd 副本
    return proc, log_path, timeout_seconds


def _kill_job(proc):
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    except ProcessLookupError:
        return
    for _ in range(10):
        if proc.poll() is not None:
            return
        time.sleep(0.5)
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except ProcessLookupError:
        pass


def _collect_secrets(cfg: dict):
    vals = []
    llm = cfg.get("llm", {})
    vals += [llm.get("api_key")]
    vol = cfg.get("narration", {}).get("tts", {}).get("volcano", {})
    vals += [vol.get("access_token"), vol.get("api_key"), vol.get("app_id")]
    return [v for v in vals if v]


RUNNER = JobRunner(_SAVED_PROXY, secrets=_collect_secrets(_raw_config()))


def _deep_merge(base: dict, extra: dict) -> dict:
    out = copy.deepcopy(base)

    def _m(d, s):
        for k, v in s.items():
            if isinstance(v, dict) and isinstance(d.get(k), dict):
                _m(d[k], v)
            else:
                d[k] = v
    _m(out, extra)
    return out


# ── Tab: 运行 / 调试 ──────────────────────────────────────────────

def _build_run_tab():
    D = _raw_config().get("narration", {}).get("explain", {})

    # ── 配置区 ──
    with gr.Row():
        with gr.Column(scale=1):
            with gr.Group(elem_classes="card"):
                gr.Markdown("### ① 选择剧集", elem_classes="card-title")
                videos = gr.CheckboxGroup(label="剧集列表（可多选，批量顺序处理）",
                                          choices=st.input_videos(), interactive=True)
                with gr.Row():
                    refresh_vid = gr.Button("刷新", size="sm")
                    sel_all = gr.Button("全选", size="sm")
                    sel_none = gr.Button("清空", size="sm")
        with gr.Column(scale=1):
            with gr.Group(elem_classes="card"):
                gr.Markdown("### ② 剪辑参数", elem_classes="card-title")
                mode = gr.Radio(["picture_led", "narration_led", "remix"],
                                value=D.get("mode", "picture_led"), label="剪辑模式")
                with gr.Row():
                    cut_style = gr.Dropdown(["bridge", "skeleton", "pure"],
                                            value=D.get("cut_style", "bridge"),
                                            label="选材方案")
                    target_minutes = gr.Slider(2, 15, value=D.get("target_minutes", 7),
                                               step=0.5, label="目标时长 (分钟)")
                with gr.Accordion("高级选项", open=False):
                    asr_only = gr.Checkbox(
                        value=False, label="仅转写（只跑语音识别就停）",
                        info="批量准备转写：出带 SPEAKER 的转写即停，不花 LLM/TTS。")
                    force_steps = gr.CheckboxGroup(
                        [zh for zh, _ in STEP_LABELS], value=[], label="强制重做",
                        info="只勾「想改变结果的最上游一步」，下游自动跟着重做；全不勾=智能缓存。")
                    gr.Markdown(
                        "| 勾选 | 作用 | 何时勾 |\n"
                        "|---|---|---|\n"
                        "| 语音识别 | 音频→带说话人转写 | 换了视频 / ASR 参数 |\n"
                        "| 剧情分析 | 挑精彩节拍 | **最常用**：想重选片、改时长/模式 |\n"
                        "| 画面备注 | VLM 看画面出备注 | 换角色参考图时 |\n"
                        "| 解说生成 | 写解说 + TTS | 只对解说/配音不满意 |\n"
                        "| 影评评论 | remix 三色弹幕 | 想换一批评论 |\n"
                        "| 裁剪拼接 | 按节拍剪 trimmed | 一般自动 |\n"
                        "| 混音 | 配音压进原声 | 改混音参数 |\n"
                        "| 字幕渲染 | 缩放+字幕+标题 | 一般自动 |\n"
                        "\n上游重做后下游指纹自动失效，无需逐个勾选。想整体重来 → 只勾「剧情分析」。")
                gr.Markdown("其余参数在「设置」页调整（仅本次生效 / 保存为默认）。")

    # ── 动作条 ──
    with gr.Group(elem_classes="actionbar"):
        with gr.Row():
            run_btn = gr.Button("运行管线", variant="primary", size="lg", scale=1)
            stop_btn = gr.Button("停止全部", variant="stop", size="sm", scale=0)
            log_toggle = gr.Button("查看运行日志", size="sm", scale=0)

    # ── 结果区 ──
    with gr.Row():
        with gr.Column(scale=1):
            with gr.Group(elem_classes="card"):
                gr.Markdown("### 任务进度", elem_classes="card-title")
                progress = gr.HTML(_render_progress("", None, 0))
                queue_box = gr.Textbox(label="任务队列", lines=4, interactive=False)
        with gr.Column(scale=1):
            with gr.Group(elem_classes="card"):
                gr.Markdown("### 成片预览", elem_classes="card-title")
                with gr.Row():
                    out_list = gr.Dropdown(label="选择成片", choices=[], scale=3)
                    refresh_out = gr.Button("刷新", size="sm", scale=1)
                prev_video = gr.Video(label="预览", height=340)
                out_zip = gr.File(label="下载选中成片", height=70)

    # ── 日志面板（默认隐藏，按钮切换）──
    log_open = gr.State(False)
    with gr.Group(elem_classes="card", visible=False) as log_panel:
        log_box = gr.Textbox(label="运行日志", lines=18, max_lines=500, autoscroll=True)

    def toggle_log(open_):
        return (gr.update(value="收起日志" if not open_ else "查看运行日志"),
                gr.update(visible=not open_), not open_)

    def do_run(vids, md, cs, tmin, force_labels, only_asr):
        if not vids:
            yield ("<div class='pp-block'><span style='color:#f87171;font-size:14px'>"
                   "请先勾选至少一集视频</span></div>",
                   RUNNER.queue_view(), "", gr.update(), gr.update())
            return
        run_ov = {"narration": {"explain": {
            "mode": md, "cut_style": cs, "target_minutes": tmin}}}
        overrides = _deep_merge(sc.SESSION_OVERRIDES, run_ov)
        force_keys = [LABEL_TO_KEY[l] for l in force_labels]
        # 仅转写：ASR 之后的步骤全部 skip（pipeline 里 skip 优先于 force）
        skip_keys = ([k for k, _l, _z, _f, can_s in STEPS if can_s and k != "asr"]
                     if only_asr else [])
        # 未勾的步骤走智能缓存（不再有独立「跳过」）；每选中一集提交一个任务，单 worker 顺序执行
        jids = [RUNNER.submit(Path(v).stem, overrides, force_keys, skip_keys) for v in vids]

        def _active():
            return [j for j in jids if RUNNER.is_active(j)]

        while _active():
            # 展示当前正在跑的那一集的进度（没有就展示第一个）
            running = next((j for j in jids if RUNNER.get(j).status == "running"), None)
            cur = RUNNER.get(running or jids[0])
            yield (_render_progress(cur.log, cur.video_name, cur.started_at),
                   RUNNER.queue_view(), cur.log, gr.update(), gr.update())
            time.sleep(0.4)
        # 收尾：拼各集结果概要 + 刷新成片列表
        tail = "\n".join(f"{RUNNER.get(j).video_name}: {RUNNER.get(j).status}" for j in jids)
        last = RUNNER.get(jids[-1])
        outs = st.all_output_mp4s()
        yield (_render_progress(last.log, last.video_name, last.started_at),
               RUNNER.queue_view(),
               last.log + "\n" + "═" * 40 + "\n批量完成：\n" + tail,
               gr.update(choices=outs, value=outs[0] if outs else None), gr.update())

    def _preview(sel):
        p = REPO_ROOT / "output" / sel if sel else None
        return (str(p), str(p)) if p and p.exists() else (None, None)

    refresh_vid.click(lambda: gr.update(choices=st.input_videos()), None, videos)
    sel_all.click(lambda: gr.update(value=st.input_videos()), None, videos)
    sel_none.click(lambda: gr.update(value=[]), None, videos)
    refresh_out.click(lambda: gr.update(choices=st.all_output_mp4s()), None, out_list)
    out_list.change(_preview, out_list, [prev_video, out_zip])
    log_toggle.click(toggle_log, log_open, [log_toggle, log_panel, log_open])
    ev = run_btn.click(do_run,
                       [videos, mode, cut_style, target_minutes, force_steps, asr_only],
                       [progress, queue_box, log_box, out_list, out_zip])
    stop_btn.click(lambda: (RUNNER.stop_all(), RUNNER.queue_view())[1], None, queue_box,
                   cancels=[ev])


# ── Tab: 集状态 ───────────────────────────────────────────────────

def _build_status_tab():
    with gr.Group(elem_classes="card"):
        gr.Markdown("### 集状态", elem_classes="card-title")
        gr.Markdown("每集各阶段产物是否已生成。")
        df = gr.Dataframe(headers=st.STATUS_HEADERS, value=st.episode_rows(),
                          interactive=False, wrap=True)
        gr.Button("刷新", size="sm").click(lambda: gr.update(value=st.episode_rows()), None, df)


# ── Tab: 剧情背景 ─────────────────────────────────────────────────

def _build_background_tab():
    with gr.Group(elem_classes="card"):
        gr.Markdown("### 剧情背景", elem_classes="card-title")
        gr.Markdown(
            "整篇 markdown 编辑：**人物介绍**段（标题含「人物」）每集都会喂给 LLM 校正人名；"
            "**分集剧情**段用 `### 第N集：…` 标题，按集精准抽取。改完保存，重跑分析/解说即生效。")
        content0, info0 = bg.load_background()
        path_info = gr.Markdown(info0)
        editor = gr.Code(value=content0, language="markdown", label="背景 markdown", lines=28)
        msg = gr.Textbox(label="结果", lines=2, interactive=False)
        with gr.Row():
            reload_btn = gr.Button("重新载入", size="sm")
            save_btn = gr.Button("保存", variant="primary")
        with gr.Accordion("按集预览（实际抽到的该集背景）", open=False):
            with gr.Row():
                ep_num = gr.Number(value=1, label="集数", precision=0, scale=1)
                prev_btn = gr.Button("预览抽取结果", scale=1)
            ep_preview = gr.Textbox(label="该集实际背景", lines=12, interactive=False)

        def _reload():
            c, i = bg.load_background()
            return c, i
        reload_btn.click(_reload, None, [editor, path_info])
        save_btn.click(bg.save_background, editor, msg)
        prev_btn.click(bg.preview_episode, [editor, ep_num], ep_preview)


# ── Tab: 设置 ─────────────────────────────────────────────────────

def _build_settings_tab():
    with gr.Group(elem_classes="card"):
        gr.Markdown("### 设置", elem_classes="card-title")
        gr.Markdown("表单改常用项，「高级编辑」改整篇 YAML。密钥留空=不改；"
                    "「仅本次生效」不写文件，「保存为默认」写入 config.yaml(自动备份)。")
        with gr.Tabs():
            with gr.TabItem("表单"):
                comps, paths, types = sc.build_form(_raw_config())
                msg = gr.Textbox(label="结果", lines=3, interactive=False)
                with gr.Row():
                    once_btn = gr.Button("仅本次生效", variant="secondary")
                    save_btn = gr.Button("保存为默认（写 config.yaml）", variant="primary")
                once_btn.click(lambda *v: sc.apply_once(paths, types, v), comps, msg)
                save_btn.click(lambda *v: sc.save_to_yaml(paths, types, v), comps, msg)
            with gr.TabItem("高级编辑 (YAML)"):
                code = gr.Code(value=sc.load_yaml_text(), language="yaml", label="config.yaml")
                ymsg = gr.Textbox(label="结果", lines=2, interactive=False)
                with gr.Row():
                    reload_btn = gr.Button("重新载入", size="sm")
                    ysave_btn = gr.Button("保存整篇", variant="primary")
                reload_btn.click(lambda: sc.load_yaml_text(), None, code)
                ysave_btn.click(sc.save_yaml_text, code, ymsg)


# ── Tab: 历史日志 ─────────────────────────────────────────────────

def _build_logs_tab():
    with gr.Group(elem_classes="card"):
        gr.Markdown("### 历史日志", elem_classes="card-title")
        gr.Markdown("logs/webui/ 下按次落盘的运行日志，密钥已脱敏。")
        with gr.Row():
            sel = gr.Dropdown(label="选择日志", choices=list_history_logs(), scale=3)
            refresh = gr.Button("刷新", size="sm", scale=1)
        content = gr.Textbox(label="内容", lines=26, max_lines=600)
        refresh.click(lambda: gr.update(choices=list_history_logs()), None, sel)
        sel.change(read_history_log, sel, content)


# ── Tab: 封面图 ───────────────────────────────────────────────────

def _build_cover_tab():
    from media import run_ffmpeg
    import tempfile

    with gr.Group(elem_classes="card"):
        gr.Markdown("### 封面图", elem_classes="card-title")
        gr.Markdown("从成片截帧 + 叠加标题生成封面。")
        cover_video = gr.Dropdown(label="选择成片", choices=[])
        gr.Button("刷新列表", size="sm").click(
            lambda: gr.update(choices=st.all_output_mp4s()), None, cover_video)
        with gr.Row():
            with gr.Column():
                capture_time = gr.Slider(0, 600, value=30, step=1, label="截图时间点 (秒)")
                capture_btn = gr.Button("截图", variant="primary")
            with gr.Column():
                title_text = gr.Textbox(value="精彩片段", label="标题文字")
                title_color = gr.Dropdown(["white", "yellow", "red", "gold", "orange"],
                                          value="orange", label="文字颜色")
                title_size = gr.Slider(24, 80, value=48, step=2, label="字号")
                overlay_btn = gr.Button("叠加文字", variant="secondary")
        preview = gr.Image(label="封面预览", height=400)

        def cap(video, sec):
            if not video:
                return None
            src = REPO_ROOT / "output" / video
            out = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
            r = run_ffmpeg(["-ss", str(sec), "-i", str(src), "-vframes", "1", "-q:v", "2", out.name])
            return out.name if r.returncode == 0 else None

        def overlay(img, t, c, s):
            if not img:
                return None
            out = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
            r = run_ffmpeg(["-i", img, "-vf",
                            f"drawtext=text='{t}':fontsize={s}:fontcolor={c}:"
                            f"x=(w-text_w)/2:y=h*0.85:fontfile=fonts/wqy-microhei.ttc",
                            "-q:v", "2", out.name])
            return out.name if r.returncode == 0 else None

        capture_btn.click(cap, [cover_video, capture_time], preview)
        overlay_btn.click(overlay, [preview, title_text, title_color, title_size], preview)


# ── Tab: 人脸标注 ─────────────────────────────────────────────────
# 注意:候选行必须用 @gr.render 动态生成。此前用 20 个隐藏 Group + 批量
# gr.update(visible=True) 的方案会触发 gradio 前端(Svelte)无限反应循环,
# 浏览器渲染线程 1 秒内硬死锁(6.5.1/6.10/6.14 均复现,已用无头浏览器矩阵定位:
# 隐藏→可见的批量翻转是唯一触发点,值更新/pending 标记/常驻可见均安全)。


def _crops_html(crops, frames, height=110):
    tags = []
    for c, f in zip(crops, frames):
        tags.append(
            f'<a href="{_file_url(f)}" target="_blank" title="点击看整帧上下文">'
            f'<img src="{_file_url(c)}" loading="lazy" decoding="async" '
            f'style="height:{height}px;margin:0;border-radius:6px"></a>')
    if not tags:
        return "<i>无</i>"
    # flex 流式排列;不用默认 prose 样式(会把 img 变成块级独占一行)
    return ('<div style="display:flex;flex-wrap:wrap;gap:4px;align-items:flex-start">'
            + "".join(tags) + "</div>")


def _build_faces_tab():
    from config import load_config
    from faces import FaceEngine

    gr.Markdown("### 人脸标注", elem_classes="card-title")
    gr.Markdown(
        "**①** 生成人物候选（各集说话时刻抽帧→检脸→聚类）→ **②** 每格是同一个人，"
        "填真名（路人留空）→ **③** 保存人脸库（自动写 `data/cast/` 参考图）。\n"
        "> 建库后：画面备注(VLM)的在场人物直接出真名，不再靠模型隔空对脸猜。\n"
        "> 同一人拆成两格就填同一个名字（自动合并）；点缩略图可看整帧上下文。\n"
        "> 🔒=已入库人物自动认出（预填名字，改名会覆盖库里的）；⚠️=该簇纯度低可能混人，重点核对。")
    fc_status = gr.Textbox(label="状态", lines=2, interactive=False)
    with gr.Row():
        fgen_btn = gr.Button("① 生成人物候选", variant="primary")
        freload_btn = gr.Button("↻ 重载已有候选", variant="secondary")
    fcands = gr.State([])

    @gr.render(inputs=[fcands])
    def _face_rows(cands):
        if not cands:
            gr.Markdown("_暂无候选。点「① 生成人物候选」，已生成过则点「↻ 重载已有候选」。_")
            return
        ids, boxes = [], []
        n_cols = 5
        for start in range(0, len(cands), n_cols):
            with gr.Row(equal_height=False):
                row_cands = cands[start:start + n_cols]
                for c in row_cands:
                    purity = float(c.get("purity", 1.0) or 0)
                    warn = " ⚠️" if purity < 0.45 else ""
                    known = c.get("name", "")
                    tag = f" · 🔒{known}" if known else ""
                    with gr.Column(scale=1, min_width=180):
                        with gr.Group():
                            n_prof = int(c.get("n_profile", 0) or 0)
                            prof = f" · 侧脸 {n_prof}" if n_prof else ""
                            gr.Markdown(f"**人物 #{c['face_id']}**{tag} · 出现 {c['count']} 次"
                                        f"{prof} · 第 {'、'.join(c.get('episodes', []))} 集{warn}")
                            gr.HTML(_crops_html(c.get("crops", []),
                                                c.get("frames", []), height=72))
                            # 锚点簇(已在人脸库里的人)预填名字,确认即可
                            nm = gr.Textbox(label="真名（留空=忽略）", value=known,
                                            interactive=True,
                                            key=f"face-nm-{c['face_id']}")
                        ids.append(c["face_id"]); boxes.append(nm)
                # 末行补空列,保持 5 等分列宽不被拉伸
                for _ in range(n_cols - len(row_cands)):
                    gr.Column(scale=1, min_width=180)
        fsave_btn = gr.Button("③ 保存人脸库(+cast参考图)", variant="primary")

        def fc_save(*nms):
            name_map = {fid: str(nm).strip() for fid, nm in zip(ids, nms)
                        if str(nm or "").strip()}
            if not name_map:
                return "没有填任何名字。"
            eng = FaceEngine(load_config(str(CONFIG_FILE)))
            eng.build_gallery(name_map)
            return (f"已保存 {len(set(name_map.values()))} 人到 {eng.gallery_file}，"
                    f"参考图写入 {eng.cast_dir}/。画面备注(VLM)会用这些真名指认在场人物。")

        fsave_btn.click(fc_save, boxes, fc_status)

    def _load_face_cands():
        return FaceEngine(load_config(str(CONFIG_FILE))).load_manifest().get("clusters", [])

    def fc_reload():
        try:
            cands = _load_face_cands()
        except Exception as e:
            return f"读取候选清单失败：{e}", []
        if not cands:
            return "暂无候选，请点「生成人物候选」。", []
        return (f"↻ 已从清单重载。共 {len(cands)} 个人物候选。"
                "每格填真名（路人留空），然后保存。"), cands

    freload_btn.click(fc_reload, None, [fc_status, fcands])

    def fc_generate():
        """生成器：立即反馈 + 每 3 秒心跳（保住 SSE 流不被代理闷死）。
        心跳只更新状态框，State 用 gr.skip() 跳过，避免反复触发重绘。"""
        try:
            proc, log_path, timeout_s = _start_face_job()
        except OSError as e:
            yield f"无法启动人脸子进程：{e}", gr.skip()
            return
        yield (f"⏳ 人脸子进程已启动（日志 logs/webui/{log_path.name}），"
               "页面其他功能不受影响…"), gr.skip()
        t0 = time.time()
        while proc.poll() is None:
            waited = int(time.time() - t0)
            if waited > timeout_s:
                _kill_job(proc)
                yield f"❌ 超过 {timeout_s}s 已终止；日志 logs/webui/{log_path.name}", gr.skip()
                return
            yield (f"⏳ 生成中 {waited}s…（faces.device=cpu 时 1-3 分钟；日志 "
                   f"logs/webui/{log_path.name}）"), gr.skip()
            time.sleep(3)
        if proc.returncode != 0:
            yield f"❌ 子进程失败(exit {proc.returncode})；日志 logs/webui/{log_path.name}", gr.skip()
            return
        try:
            cands = _load_face_cands()
        except Exception as e:
            yield f"✅ 生成完成，但读取候选清单失败：{e}", gr.skip()
            return
        yield (f"✅ 生成完成。共 {len(cands)} 个人物候选。"
               "每格填真名（路人留空），然后保存。"), cands

    fgen_btn.click(fc_generate, None, [fc_status, fcands])


def _file_url(p):
    return "gradio_api/file=" + urllib.parse.quote(str(Path(p).resolve()))


# ── 组装 ──────────────────────────────────────────────────────────

def build_app():
    with gr.Blocks(title="电影解说工作台") as app:
        gr.HTML(HEADER_HTML)
        with gr.Tabs():
            with gr.TabItem("运行 / 调试"):
                _build_run_tab()
            with gr.TabItem("集状态"):
                _build_status_tab()
            with gr.TabItem("剧情背景"):
                _build_background_tab()
            with gr.TabItem("设置"):
                _build_settings_tab()
            with gr.TabItem("历史日志"):
                _build_logs_tab()
            with gr.TabItem("封面图"):
                _build_cover_tab()
            with gr.TabItem("人脸标注"):
                # 注意:不给 tab 挂 select 回调——切页导航必须零成本,重载走显式按钮
                _build_faces_tab()
    return app


def main():
    (REPO_ROOT / "input").mkdir(exist_ok=True)
    (REPO_ROOT / "output").mkdir(exist_ok=True)
    app = build_app()
    app.queue(default_concurrency_limit=4).launch(
        server_name="0.0.0.0", server_port=6006, share=False,
        theme=_build_theme(), css=CSS,
        allowed_paths=[str((REPO_ROOT / "output").resolve()),
                       str((REPO_ROOT / "input").resolve())])


if __name__ == "__main__":
    main()
