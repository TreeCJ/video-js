"""ASR：阿里百炼 Paraformer 录音文件识别（云端转写 + 云端说话人分离）。

替代原来的 WhisperX（faster-whisper + 词级对齐 + pyannote 分离）本地链路：
- 抽音频：ffmpeg → 16kHz 单声道 wav（paraformer 分离要求单声道）
- 上传：dashscope 临时 OSS（OssUtils.upload），拿 oss:// URL
- 转写：Transcription.async_call + 轮询，diarization_enabled 出句级 speaker_id
- 落盘：transcript JSON + SRT，格式与旧版一致，下游零改动

每个 segment 多出 `speaker` 字段（SPEAKER_00/01…，由云端 speaker_id 映射），
向后兼容旧的 {start,end,text} 结构。
"""
import json
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import List, Tuple

from media import run_ffmpeg

_PROXY_KEYS = ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY",
               "all_proxy", "ALL_PROXY")


def seconds_to_srt_time(seconds: float) -> str:
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


@contextmanager
def _no_proxy():
    """临时摘掉代理环境变量。dashscope 的 OSS 上传/查询走 aiohttp/requests，
    会读 http(s)_proxy；HF 下载的 SOCKS 代理会把这些端点也拖进代理导致失败。
    与 llm.make_httpx_client 同理：百炼端点要直连。"""
    old = {k: os.environ.pop(k) for k in _PROXY_KEYS if k in os.environ}
    try:
        yield
    finally:
        os.environ.update(old)


class ASR:
    """百炼 Paraformer 云端转写 + 说话人分离。"""

    def __init__(self, model: str = "paraformer-v2", api_key: str = None,
                 language: str = "zh", language_hints: List[str] = None,
                 diarize: bool = True):
        self.logger = None
        self.model = model
        self.api_key = api_key or os.environ.get("DASHSCOPE_API_KEY") or ""
        self.language = language
        self.language_hints = language_hints or ["zh", "en"]
        self.diarize = diarize

    def set_logger(self, logger):
        self.logger = logger

    def _log(self, msg):
        if self.logger:
            self.logger.info(msg)

    # ── 结果映射（纯函数，可测）────────────────────────────────────
    @staticmethod
    def _map_segments(data: dict, diarize: bool = True) -> List[dict]:
        """百炼转录结果 → transcript segments。

        paraformer 天然按「句」切分，一句即一个说话人的完整话，无需再按词级
        拆分跨人长段。`speaker_id` 是句级整数（0 起），映射成 SPEAKER_xx。
        """
        segments = []
        for tr in (data.get("transcripts") or []):
            for s in (tr.get("sentences") or []):
                text = (s.get("text") or "").strip()
                if not text:
                    continue
                seg = {
                    "start": round(float(s.get("begin_time", 0)) / 1000.0, 3),
                    "end": round(float(s.get("end_time", 0)) / 1000.0, 3),
                    "text": text,
                }
                if diarize and s.get("speaker_id") is not None:
                    seg["speaker"] = f"SPEAKER_{int(s['speaker_id']):02d}"
                segments.append(seg)
        return segments

    # ── 云端链路 ──────────────────────────────────────────────────
    def _extract_audio(self, video_path: str, tmp_dir: Path) -> str:
        wav = tmp_dir / "audio.wav"
        self._log("Extracting 16kHz mono audio ...")
        run_ffmpeg(["-i", video_path, "-vn", "-ac", "1", "-ar", "16000",
                    "-c:a", "pcm_s16le", str(wav)], check=True)
        return str(wav)

    def _transcribe_cloud(self, wav_path: str) -> dict:
        from dashscope.audio.asr import Transcription
        from dashscope.utils.oss_utils import OssUtils

        if not self.api_key:
            raise RuntimeError("未配置百炼 API key（asr.api_key 或 DASHSCOPE_API_KEY）")

        kwargs = {"diarization_enabled": True} if self.diarize else {}
        if self.model == "paraformer-v2" and self.language_hints:
            kwargs["language_hints"] = self.language_hints
        headers = {"X-DashScope-OssResourceResolve": "enable"}

        with _no_proxy():
            self._log("Uploading audio to DashScope OSS ...")
            oss_url, _ = OssUtils.upload(
                model=self.model, file_path=wav_path, api_key=self.api_key)
            self._log(f"Submitting {self.model} (diarize={self.diarize}) ...")
            resp = Transcription.async_call(
                model=self.model, file_urls=[oss_url],
                api_key=self.api_key, headers=headers, **kwargs)
            if resp.status_code != 200 or not (resp.output or {}).get("task_id"):
                raise RuntimeError(f"提交识别任务失败: code={resp.code} msg={resp.message}")
            task_id = resp.output["task_id"]
            self._log(f"Task submitted: {task_id}, waiting ...")
            out = Transcription.wait(task=task_id, api_key=self.api_key,
                                     headers=headers).output

        if out.get("task_status") != "SUCCEEDED":
            raise RuntimeError(f"识别任务失败: status={out.get('task_status')} "
                               f"code={out.get('code')}")

        results = out.get("results") or []
        if not results:
            raise RuntimeError("识别结果为空")
        first = results[0]
        url = (first.get("transcription_url")
               or (first.get("output") or {}).get("transcription_url"))
        if not url:
            raise RuntimeError("结果里没有 transcription_url")
        return self._download_result(url)

    def _download_result(self, url: str) -> dict:
        import httpx
        with _no_proxy(), httpx.Client(trust_env=False, follow_redirects=True,
                                       timeout=120.0) as client:
            resp = client.get(url)
            resp.raise_for_status()
            return resp.json()

    def transcribe(self, video_path: str, output_dir: str = "output") -> Tuple[str, str]:
        self._log(f"Starting Paraformer transcription for: {video_path}")
        self._log(f"model={self.model} lang={self.language} diarize={self.diarize}")

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        base_name = Path(video_path).stem

        with tempfile.TemporaryDirectory(prefix="asr_") as tmp:
            wav = self._extract_audio(video_path, Path(tmp))
            data = self._transcribe_cloud(wav)

        segments = self._map_segments(data, self.diarize)
        n_spk = len({s["speaker"] for s in segments if s.get("speaker")})
        if self.diarize:
            self._log(f"Diarization done: {n_spk} distinct speakers")

        transcript_path = output_path / f"{base_name}_transcript.json"
        srt_path = output_path / f"{base_name}_full.srt"

        with open(transcript_path, "w", encoding="utf-8") as f:
            json.dump({"segments": segments}, f, ensure_ascii=False, indent=2)
        self._log(f"Transcript saved to: {transcript_path}")

        with open(srt_path, "w", encoding="utf-8") as f:
            for i, seg in enumerate(segments, 1):
                spk = f"[{seg['speaker']}] " if seg.get("speaker") else ""
                f.write(f"{i}\n")
                f.write(f"{seconds_to_srt_time(seg['start'])} --> "
                        f"{seconds_to_srt_time(seg['end'])}\n")
                f.write(f"{spk}{seg['text']}\n\n")
        self._log(f"SRT subtitle saved to: {srt_path}")
        self._log(f"Total segments: {len(segments)}")

        return str(transcript_path), str(srt_path)


def transcribe_video(video_path: str, config, output_dir: str = "output") -> Tuple[str, str]:
    asr_cfg = config.asr
    asr = ASR(
        model=asr_cfg.get("model", "paraformer-v2"),
        api_key=asr_cfg.get("api_key"),
        language=asr_cfg.get("language", "zh"),
        language_hints=asr_cfg.get("language_hints"),
        diarize=asr_cfg.get("diarize", True),
    )
    asr.set_logger(config.get_logger("ASR"))
    return asr.transcribe(video_path, output_dir)


if __name__ == "__main__":
    import sys
    from config import load_config

    config_path = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
    config = load_config(config_path)
    asr = ASR(
        model=config.asr.get("model", "paraformer-v2"),
        api_key=config.asr.get("api_key"),
        language=config.asr.get("language", "zh"),
        language_hints=config.asr.get("language_hints"),
        diarize=config.asr.get("diarize", True),
    )
    asr.set_logger(config.get_logger("ASR"))

    import glob
    input_dir = config.video.get("input_dir", "input")
    videos = glob.glob(f"{input_dir}/*.mp4") + glob.glob(f"{input_dir}/*.avi")
    if not videos:
        print("No video files found")
        sys.exit(1)
    transcript_path, srt_path = asr.transcribe(videos[0])
    print(f"Transcript: {transcript_path}")
    print(f"SRT: {srt_path}")
