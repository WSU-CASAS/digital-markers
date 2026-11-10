import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import digital_markers.audio.markers as audio_markers
from digital_markers import AudioRecording, MarkerResult, ParticipantData, read_audio_directory
from digital_markers.audio import compute_audio_markers, write_audio_transcripts


def test_audio_recording_accepts_path_and_sample_backed_inputs(tmp_path):
    source = tmp_path / "voice.m4a"
    source.touch()

    path_recording = read_audio_directory(tmp_path)
    sample_recording = AudioRecording(
        filename="memory.wav",
        samples=np.zeros((160, 2), dtype=np.float32),
        sample_rate_hz=16_000,
    )

    assert len(path_recording) == 1
    assert path_recording[0].filename == "voice.m4a"
    assert path_recording[0].path == source
    assert sample_recording.samples.shape == (160, 2)


def test_read_audio_directory_scans_only_direct_files(tmp_path):
    (tmp_path / "first.wav").touch()
    (tmp_path / "ignore.txt").touch()
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "nested.wav").touch()

    assert [recording.filename for recording in read_audio_directory(tmp_path)] == ["first.wav"]


def test_audio_samples_normalize_integer_pcm():
    recording = AudioRecording(
        filename="memory.wav",
        samples=np.array([-32768, 0, 32767], dtype=np.int16),
        sample_rate_hz=16_000,
    )

    samples = audio_markers._audio_samples(recording)

    assert samples.dtype == np.float32
    assert samples[0] == -1.0
    assert samples[-1] == pytest.approx(1.0, abs=1 / 32768)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"filename": "voice.wav"}, "exactly one"),
        (
            {"filename": "voice.wav", "path": "voice.wav", "samples": np.zeros(2)},
            "exactly one",
        ),
        (
            {"filename": "folder/voice.wav", "samples": np.zeros(2), "sample_rate_hz": 16_000},
            "file name",
        ),
        (
            {"filename": "voice.wav", "samples": np.zeros(2)},
            "sample_rate_hz",
        ),
    ],
)
def test_audio_recording_rejects_invalid_sources(kwargs, message):
    with pytest.raises(ValueError, match=message):
        AudioRecording(**kwargs)


def test_compute_audio_markers_returns_participant_summary_and_transcripts(monkeypatch):
    recordings = [
        AudioRecording("voice_20260101.wav", samples=np.zeros(16), sample_rate_hz=16_000),
        AudioRecording("voice_20260102.wav", samples=np.zeros(16), sample_rate_hz=16_000),
    ]
    monkeypatch.setattr(
        audio_markers, "_create_whisper_model", lambda model, device: (object(), "cpu")
    )
    monkeypatch.setattr(audio_markers, "_TextModels", lambda *_args, **_kwargs: object())

    def analyze(recording, *_args, **_kwargs):
        return {
            "filename": recording.filename,
            "date": "2026-01-01" if "20260101" in recording.filename else "2026-01-02",
            "language": "en",
            "text": f"Transcript for {recording.filename}",
            "informative": 1,
            "words": 10,
            "speech_rate": 100.0 if "20260101" in recording.filename else 120.0,
        }

    monkeypatch.setattr(audio_markers, "_analyze_recording", analyze)
    result = compute_audio_markers(
        ParticipantData(participant_id="P001", audio=recordings),
        language="zh",
    )

    assert result.markers.loc[0, "id"] == "P001"
    assert result.markers.loc[0, "n_recordings"] == 2
    assert result.markers.loc[0, "speech_rate_mean"] == 110.0
    assert result.metadata["language"] == "zh"
    assert result.details["recordings"]["text"].tolist() == [
        "Transcript for voice_20260101.wav",
        "Transcript for voice_20260102.wav",
    ]


def test_compute_audio_markers_requires_audio_inputs():
    with pytest.raises(ValueError, match="Missing required data for audio"):
        compute_audio_markers(ParticipantData(participant_id="P001"))


def test_text_features_support_english_behavior_health_and_custom_language_patterns(monkeypatch):
    class Embedder:
        def encode(self, texts, **_kwargs):
            return np.array([[1.0, 0.0], [0.8, 0.6]])

    class Models:
        def __init__(self):
            self.embedder = Embedder()
            self.parsers = {}

        def sentiment(self, _text, _language):
            return 0.25

    monkeypatch.setitem(
        sys.modules,
        "wordfreq",
        SimpleNamespace(zipf_frequency=lambda _word, _language: 4.0),
    )
    text = "Yesterday I cooked dinner with my family at home. I had trouble carrying things."
    english_patterns = audio_markers._validate_text_patterns(None)["en"]
    english = audio_markers._text_features(
        text,
        ["Yesterday", "I", "cooked", "dinner", "with", "my", "family", "at", "home"],
        [text],
        Models(),
        language="en",
        min_words=1,
        include_behavior=True,
        include_health=True,
        patterns=english_patterns,
    )

    assert english["specificity"] == 1.0
    assert english["activity_diversity"] == 2
    assert english["activity_count"] == 2
    assert english["household"] == 1
    assert english["social"] == 1
    assert english["difficulty"] == 1
    assert english["sentiment"] == 0.25
    assert english["lexical_sophistication"] == 3.0

    custom_patterns = audio_markers._validate_text_patterns(
        {
            "xx": {
                "activities": {"food": r"\bfoo\b", "social": r"\bbar\b"},
                "health": {"memory": r"\bbaz\b"},
                "temporal": r"\byesterday\b",
                "detail": r"\bhome\b",
            }
        }
    )["xx"]
    other_language = audio_markers._text_features(
        "foo bar baz yesterday home",
        ["foo", "bar", "baz", "yesterday", "home"],
        ["foo bar baz yesterday home"],
        Models(),
        language="xx",
        min_words=1,
        include_behavior=True,
        include_health=True,
        patterns=custom_patterns,
    )
    assert other_language["activity_diversity"] == 2
    assert other_language["activity_count"] == 2
    assert other_language["social"] == 1
    assert other_language["household"] == 1
    assert other_language["memory"] == 1
    assert other_language["specificity"] == 1.0
    assert pd.isna(other_language["difficulty"])
    assert pd.isna(other_language["community"])


def test_text_pattern_validation_rejects_invalid_regular_expressions():
    with pytest.raises(ValueError, match="Invalid text pattern"):
        audio_markers._validate_text_patterns({"xx": {"health": {"memory": "["}}})


def test_write_audio_transcripts_uses_original_stems(tmp_path):
    result = MarkerResult(
        markers=pd.DataFrame(),
        details={
            "recordings": pd.DataFrame(
                {
                    "filename": ["voice_001.m4a", "journal.wav"],
                    "text": ["First transcript.", "Second transcript."],
                }
            )
        },
    )

    written = write_audio_transcripts(result, tmp_path / "transcripts")

    assert [path.name for path in written] == [
        "voice_001-transcript.txt",
        "journal-transcript.txt",
    ]
    assert written[0].read_text(encoding="utf-8") == "First transcript."
    assert written[1].read_text(encoding="utf-8") == "Second transcript."


def test_write_audio_transcripts_accepts_empty_suffix(tmp_path):
    result = MarkerResult(
        markers=pd.DataFrame(),
        details={
            "recordings": pd.DataFrame(
                {"filename": ["voice_001.m4a"], "text": ["First transcript."]}
            )
        },
    )

    written = write_audio_transcripts(result, tmp_path, suffix="")

    assert [path.name for path in written] == ["voice_001.txt"]
    assert written[0].read_text(encoding="utf-8") == "First transcript."


def test_write_audio_transcripts_refuses_colliding_names(tmp_path):
    result = MarkerResult(
        markers=pd.DataFrame(),
        details={
            "recordings": pd.DataFrame(
                {"filename": ["voice.wav", "voice.m4a"], "text": ["first", "second"]}
            )
        },
    )

    with pytest.raises(ValueError, match="collision"):
        write_audio_transcripts(result, tmp_path)
    assert list(tmp_path.iterdir()) == []
