"""分片上传后端（src/webui/upload.py）测试。"""
import io
import os
import shutil
import time
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import status as st
import upload


@pytest.fixture()
def env(tmp_path, monkeypatch):
    tmp_dir = tmp_path / "upload_tmp"
    in_dir = tmp_path / "input"
    monkeypatch.setattr(upload, "UPLOAD_TMP_DIR", tmp_dir)
    monkeypatch.setattr(st, "input_dir", lambda: in_dir)
    app = FastAPI()
    upload.register_routes(app)
    return TestClient(app), tmp_dir, in_dir


def _post_chunk(client, upload_id, name, idx, total, cs, fs, data):
    return client.post(
        "/upload/chunk",
        data={"upload_id": upload_id, "filename": name, "chunk_index": str(idx),
              "total_chunks": str(total), "chunk_size": str(cs), "file_size": str(fs)},
        files={"chunk": (name, io.BytesIO(data))})


def _post_complete(client, upload_id, name, total, cs, fs):
    return client.post(
        "/upload/complete",
        data={"upload_id": upload_id, "filename": name, "total_chunks": str(total),
              "chunk_size": str(cs), "file_size": str(fs)})


def test_roundtrip_multichunk(env, monkeypatch):
    client, tmp_dir, in_dir = env
    monkeypatch.setattr(upload, "_probe_video", lambda p: True)
    payload = b"A" * 20   # 3 片：8 + 8 + 4
    for i, data in enumerate([b"A" * 8, b"A" * 8, b"A" * 4]):
        r = _post_chunk(client, "upload-1234", "ep01.mp4", i, 3, 8, 20, data)
        assert r.status_code == 200, r.text
    r = _post_complete(client, "upload-1234", "ep01.mp4", 3, 8, 20)
    assert r.status_code == 200, r.text
    assert r.json() == {"ok": True, "name": "ep01.mp4", "size": 20}
    assert (in_dir / "ep01.mp4").read_bytes() == payload
    assert not (tmp_dir / "upload-1234").exists()          # 会话目录已清理
    assert not list(in_dir.glob("*.part"))                 # 无 .part 残片


def test_complete_reports_missing_chunk(env, monkeypatch):
    client, _, _ = env
    monkeypatch.setattr(upload, "_probe_video", lambda p: True)
    for i in range(2):   # 缺最后一片
        r = _post_chunk(client, "upload-1234", "e.mp4", i, 3, 8, 20, b"A" * 8)
        assert r.status_code == 200
    r = _post_complete(client, "upload-1234", "e.mp4", 3, 8, 20)
    assert r.status_code == 400
    assert "缺少分片" in r.json()["detail"]


def test_complete_missing_session_404(env):
    client, _, _ = env
    r = _post_complete(client, "upload-1234", "e.mp4", 1, 8, 8)
    assert r.status_code == 404


def test_chunk_rejects_wrong_byte_count(env):
    client, _, _ = env
    r = _post_chunk(client, "upload-1234", "e.mp4", 0, 2, 8, 16, b"short")
    assert r.status_code == 400
    assert "分片大小不符" in r.json()["detail"]


def test_chunk_rejects_total_mismatch(env):
    client, _, _ = env
    r = _post_chunk(client, "upload-1234", "e.mp4", 0, 3, 8, 16, b"A" * 8)  # ceil(16/8)=2，传 3 应失败
    assert r.status_code == 400


def test_chunk_rejects_bad_index(env):
    client, _, _ = env
    r = _post_chunk(client, "upload-1234", "e.mp4", 5, 2, 8, 16, b"A" * 8)
    assert r.status_code == 400


def test_rejects_bad_extension(env):
    client, _, _ = env
    r = _post_chunk(client, "upload-1234", "evil.sh", 0, 1, 8, 8, b"A" * 8)
    assert r.status_code == 400
    assert "文件类型" in r.json()["detail"]


def test_rejects_traversal_name(env):
    client, _, _ = env
    r = _post_chunk(client, "upload-1234", "../../etc/cron.d/x.mp4", 0, 1, 8, 8, b"A" * 8)
    assert r.status_code == 200  # Path.name 剥掉路径，只留文件名，不构成穿越
    assert r.json()["received"] == 0


def test_rejects_bad_upload_id(env):
    client, _, _ = env
    r = _post_chunk(client, "bad id!", "e.mp4", 0, 1, 8, 8, b"A" * 8)
    assert r.status_code == 400


def test_same_name_gets_suffix(env, monkeypatch):
    client, _, in_dir = env
    monkeypatch.setattr(upload, "_probe_video", lambda p: True)
    for uid in ("upload-0001", "upload-0002"):
        _post_chunk(client, uid, "ep01.mp4", 0, 1, 8, 8, b"A" * 8)
        r = _post_complete(client, uid, "ep01.mp4", 1, 8, 8)
        assert r.status_code == 200
    assert (in_dir / "ep01.mp4").exists()
    assert (in_dir / "ep01_1.mp4").exists()


@pytest.mark.skipif(shutil.which("ffprobe") is None, reason="ffprobe not available")
def test_probe_failure_removes_partial_file(env):
    client, _, in_dir = env
    _post_chunk(client, "upload-1234", "broken.mp4", 0, 1, 8, 8, b"garbage!")
    r = _post_complete(client, "upload-1234", "broken.mp4", 1, 8, 8)
    assert r.status_code == 422
    assert not (in_dir / "broken.mp4").exists()
    assert not list(in_dir.glob("*.part"))


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not available")
def test_real_video_passes_probe(env):
    from media import run_ffmpeg
    client, _, in_dir = env
    src = in_dir / "raw_src.mp4"
    in_dir.mkdir(parents=True, exist_ok=True)
    run_ffmpeg(["-f", "lavfi", "-i", "color=c=black:s=320x240:d=0.5", str(src)],
               check=True)
    data = src.read_bytes()
    cs = max(1, len(data))
    r = _post_chunk(client, "upload-1234", "real.mp4", 0, 1, cs, cs, data)
    assert r.status_code == 200, r.text
    r = _post_complete(client, "upload-1234", "real.mp4", 1, cs, cs)
    assert r.status_code == 200, r.text
    assert (in_dir / "real.mp4").read_bytes() == data


def test_cleanup_stale(env):
    _client, tmp_dir, in_dir = env
    old = tmp_dir / "upload-old"
    old.mkdir(parents=True)
    (old / "chunk_0").write_bytes(b"x")
    new = tmp_dir / "upload-new"
    new.mkdir()
    (new / "chunk_0").write_bytes(b"x")
    in_dir.mkdir(parents=True, exist_ok=True)
    stale_part = in_dir / "ep01.mp4.part"
    stale_part.write_bytes(b"x")
    stale_part.touch()
    old_t = time.time() - 48 * 3600
    os.utime(old, (old_t, old_t))
    os.utime(stale_part, (old_t, old_t))
    upload.cleanup_stale()
    assert not old.exists()
    assert new.exists()
    assert not stale_part.exists()
