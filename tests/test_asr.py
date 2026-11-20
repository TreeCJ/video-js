from asr import ASR


def test_map_segments_maps_speaker_id_to_speaker_label():
    """云端句级 speaker_id(整数) → SPEAKER_xx，时间戳毫秒 → 秒。"""
    data = {"transcripts": [{"sentences": [
        {"begin_time": 0, "end_time": 3840, "text": "今天天气很好。", "speaker_id": 0},
        {"begin_time": 4000, "end_time": 6000, "text": "一起去公园吧。", "speaker_id": 1},
    ]}]}

    segs = ASR._map_segments(data, diarize=True)

    assert segs == [
        {"start": 0.0, "end": 3.84, "text": "今天天气很好。", "speaker": "SPEAKER_00"},
        {"start": 4.0, "end": 6.0, "text": "一起去公园吧。", "speaker": "SPEAKER_01"},
    ]


def test_map_segments_without_diarize_omits_speaker():
    data = {"transcripts": [{"sentences": [
        {"begin_time": 0, "end_time": 1000, "text": "你好。"},
    ]}]}

    segs = ASR._map_segments(data, diarize=False)

    assert segs == [{"start": 0.0, "end": 1.0, "text": "你好。"}]
    assert "speaker" not in segs[0]


def test_map_segments_skips_empty_and_missing_transcripts():
    assert ASR._map_segments({}, diarize=True) == []
    assert ASR._map_segments({"transcripts": []}, diarize=True) == []
    assert ASR._map_segments(
        {"transcripts": [{"sentences": [{"begin_time": 0, "end_time": 10,
                                         "text": "   ", "speaker_id": 0}]}]},
        diarize=True) == []
