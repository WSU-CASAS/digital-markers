from __future__ import annotations

import re
import warnings
from collections.abc import Mapping, Sequence
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import resample_poly

from digital_markers.context import MarkerContext
from digital_markers.data.audio import AudioRecording
from digital_markers.data.participant import ParticipantData
from digital_markers.results import MarkerResult

AUDIO_FEATURES = (
    "speech_rate",
    "articulation_rate",
    "pause_mean",
    "pause_sd",
    "pause_fraction",
    "pitch_sd",
    "intensity_sd",
    "hnr",
    "cpp",
    "jitter",
    "shimmer",
    "spectral_tilt",
    *(f"mfcc_{index}" for index in range(1, 14)),
)
LANGUAGE_FEATURES = (
    "words",
    "lexical_diversity",
    "lexical_sophistication",
    "syntax_complexity",
    "specificity",
    "sentiment",
    "topic_coherence",
    "topic_shift",
    "semantic_repetition",
)
BEHAVIOR_FEATURES = (
    "activity_count",
    "activity_diversity",
    "household",
    "community",
    "social",
    "physical",
    "cognitive",
)
HEALTH_FEATURES = ("difficulty", "memory", "physical_limit", "assistance", "compensation")
DATE_PATTERN = re.compile(r"(?<!\d)(20\d{2})[-_]?([01]\d)[-_]?([0-3]\d)(?!\d)")
WORD_PATTERN = re.compile(r"[^\W\d_]+(?:['’][^\W\d_]+)*", re.UNICODE)
SENTENCE_PATTERN = re.compile(r"(?<=[.!?])\s+|\n+")
ENGLISH_ACTIVITY_PATTERNS = {
    "food": r"\b(?:cook(?:ed|ing)?|bak(?:e|ed|ing)|prepar(?:ed|ing) (?:a |the )?(?:meal|food|dinner|lunch|breakfast)|made (?:breakfast|lunch|dinner))\b",
    "housework": r"\b(?:clean(?:ed|ing)?|vacuum(?:ed|ing)?|laundry|wash(?:ed|ing)? (?:clothes|dishes)|housework|chores)\b",
    "maintenance": r"\b(?:repair(?:ed|ing)?|fix(?:ed|ing)? (?:a |the )?(?:sink|door|fence|appliance|house)|yard work|mow(?:ed|ing)?|garden(?:ed|ing)?)\b",
    "errands": r"\b(?:shopp(?:ed|ing)|grocer(?:y|ies)|errands?|went to (?:the )?(?:store|market|pharmacy|bank)|picked up (?:groceries|a prescription))\b",
    "appointments": r"\b(?:appointment|doctor(?:'s)? visit|dentist|physical therapy|clinic|hospital visit)\b",
    "outings": r"\b(?:went (?:out|to|over)|drove to|visited (?:a |the )?(?:park|museum|library|restaurant|church)|went for a (?:walk|drive)|left (?:the )?house)\b",
    "social": r"\b(?:talk(?:ed|ing)? (?:to|with)|chat(?:ted|ting)? with|call(?:ed|ing)? (?:my |a |the )?(?:friend|daughter|son|sister|brother|mother|father)|phone call|visited with|met (?:with |up with )|lunch with|dinner with|friends? over|family over)\b",
    "physical": r"\b(?:walk(?:ed|ing)?|hik(?:ed|ing)|sw(?:am|im|imming)|exercis(?:ed|ing)|gym|yoga|stretch(?:ed|ing)?|bik(?:ed|ing)|cycl(?:ed|ing)|danc(?:ed|ing)|gardening|yard work)\b",
    "cognitive": r"\b(?:read(?:ing)? (?:a |the )?(?:book|novel|article|paper)|crossword|puzzle|sudoku|board games?|card games?|knitt?(?:ed|ing)|sew(?:ed|ing)?|paint(?:ed|ing)?|stud(?:y|ied|ying)|learn(?:ed|ing)?|practic(?:ed|ing)? (?:music|piano|guitar)|play(?:ed|ing)? (?:piano|chess|guitar))\b",
}
ENGLISH_HEALTH_PATTERNS = {
    "difficulty": r"\b(?:had trouble|had difficulty|struggl(?:ed|ing)|could not (?:finish|manage|complete|do)|was unable to|hard for me to|difficult for me to|trouble (?:with|doing))\b",
    "memory": r"\b(?:forgot|forget|couldn.t remember|can.t remember|could not remember|misplaced|lost track of|had trouble remembering)\b",
    "physical_limit": r"\b(?:too tired to|fatigu(?:e|ed)|exhausted|pain(?:ful)?|sore|dizz(?:y|iness)|weakness|trouble walking|couldn.t walk|could not walk|short of breath)\b",
    "assistance": r"\b(?:helped me|help from|needed help|assisted me|someone (?:helped|drove) me|asked .{0,40} for help)\b",
    "compensation": r"\b(?:used (?:a |my )?(?:reminder|calendar|list|notes?|alarm|walker)|set (?:a |an )?(?:reminder|alarm)|wrote (?:it|that) down|made (?:a |my )?list|had to adapt|worked around|figured it out|managed on my own)\b",
}
ENGLISH_PATTERNS: dict[str, object] = {
    "activities": ENGLISH_ACTIVITY_PATTERNS,
    "health": ENGLISH_HEALTH_PATTERNS,
    "temporal": r"\b(?:this morning|this afternoon|this evening|last night|yesterday|today|tomorrow|before|after|later|earlier|at \d{1,2}(?::\d{2})?(?: ?[ap]m)?)\b",
    "detail": r"\b(?:my |our )?(?:husband|wife|daughter|son|friend|neighbor|sister|brother|mother|father|grandchild|family)\b|\b(?:at|in|to|from) (?:the |my |a )?(?:store|park|restaurant|church|library|home|house|garden|hospital|office|school|beach|gym|market|bank|clinic)\b",
}
ACTIVITY_DOMAINS = {
    "household": ("food", "housework", "maintenance"),
    "community": ("errands", "appointments", "outings"),
    "social": ("social",),
    "physical": ("physical",),
    "cognitive": ("cognitive",),
}
ACTIVITY_CATEGORIES = tuple(ENGLISH_ACTIVITY_PATTERNS)


def _finite(value: object) -> float:
    try:
        return float(value) if np.isfinite(value) else np.nan
    except (TypeError, ValueError):
        return np.nan


def _date_from_filename(filename: str) -> str | None:
    match = DATE_PATTERN.search(filename)
    if match is None:
        return None
    try:
        return date(
            int(match.group(1)),
            int(match.group(2)),
            int(match.group(3)),
        ).isoformat()
    except ValueError:
        return None


def _audio_samples(recording: AudioRecording, sample_rate: int = 16_000) -> np.ndarray:
    if recording.path is not None:
        from faster_whisper.audio import decode_audio

        return np.asarray(
            decode_audio(str(recording.path), sampling_rate=sample_rate), dtype=np.float32
        )

    raw_samples = np.asarray(recording.samples)
    if np.issubdtype(raw_samples.dtype, np.integer):
        limits = np.iinfo(raw_samples.dtype)
        if np.issubdtype(raw_samples.dtype, np.unsignedinteger):
            midpoint = (limits.max + 1) / 2
            samples = (raw_samples.astype(np.float32) - midpoint) / midpoint
        else:
            scale = max(abs(limits.min), limits.max)
            samples = raw_samples.astype(np.float32) / scale
    else:
        samples = raw_samples.astype(np.float32)
    if samples.ndim == 2:
        samples = samples.mean(axis=1)
    if recording.sample_rate_hz != sample_rate:
        from math import gcd

        divisor = gcd(recording.sample_rate_hz, sample_rate)
        samples = resample_poly(
            samples,
            sample_rate // divisor,
            recording.sample_rate_hz // divisor,
        ).astype(np.float32)
    return samples


def _asr_features(
    source: str | np.ndarray,
    model: object,
    *,
    language: str,
    beam_size: int,
) -> dict[str, object]:
    segments_iter, info = model.transcribe(  # type: ignore[attr-defined]
        source,
        language=None if language == "auto" else language,
        task="transcribe",
        beam_size=beam_size,
        best_of=5,
        patience=1,
        temperature=(0.0, 0.2, 0.4, 0.6, 0.8),
        compression_ratio_threshold=2.4,
        log_prob_threshold=-1.0,
        no_speech_threshold=0.6,
        condition_on_previous_text=False,
        word_timestamps=True,
        vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 700, "speech_pad_ms": 250},
        hallucination_silence_threshold=2.0,
    )
    segments = list(segments_iter)
    pieces: list[dict[str, object]] = []
    words: list[dict[str, object]] = []
    tokens: list[str] = []
    word_times: list[tuple[float, float]] = []
    logprobs: list[float] = []
    for segment in segments:
        pieces.append(
            {"start": float(segment.start), "end": float(segment.end), "text": segment.text.strip()}
        )
        if np.isfinite(segment.avg_logprob):
            logprobs.append(float(segment.avg_logprob))
        for word in segment.words or []:
            if word.word.strip():
                tokens.append(word.word.strip())
            if word.start is not None and word.end is not None and word.end > word.start:
                words.append(
                    {"start": float(word.start), "end": float(word.end), "text": word.word}
                )
                word_times.append((float(word.start), float(word.end)))
    return {
        "text": " ".join(str(piece["text"]) for piece in pieces).strip(),
        "segments": pieces,
        "words": words,
        "tokens": tokens,
        "word_times": word_times,
        "language": info.language,
        "language_probability": _finite(info.language_probability),
        "asr_avg_logprob": _finite(np.mean(logprobs)) if logprobs else np.nan,
    }


def _intervals_union(intervals: Sequence[tuple[float, float]]) -> list[tuple[float, float]]:
    merged: list[tuple[float, float]] = []
    for start, end in sorted((start, end) for start, end in intervals if end > start):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
        else:
            merged.append((start, end))
    return merged


def _spectral_cpp(samples: np.ndarray, sample_rate: int) -> float:
    hop = int(0.01 * sample_rate)
    frame = int(0.04 * sample_rate)
    nfft = 2048
    if len(samples) < frame:
        return np.nan
    values = []
    for start in range(0, len(samples) - frame + 1, hop):
        clip = samples[start : start + frame]
        if np.sqrt(np.mean(clip * clip)) < 0.008 * np.max(np.abs(samples) + 1e-8):
            continue
        spectrum = np.fft.rfft(clip * np.hamming(len(clip)), n=nfft)
        cepstrum = np.fft.irfft(np.log(np.maximum(np.abs(spectrum), 1e-10)), n=nfft)
        quefrency = np.arange(len(cepstrum)) / sample_rate
        baseline = (quefrency >= 0.001) & (quefrency <= 0.016)
        peak = (quefrency >= 0.0033) & (quefrency <= 0.016)
        if baseline.sum() < 5 or peak.sum() < 2:
            continue
        slope, intercept = np.polyfit(quefrency[baseline], cepstrum[baseline], 1)
        values.append(
            np.max(cepstrum[peak] - (slope * quefrency[peak] + intercept)) * 20 / np.log(10)
        )
    return _finite(np.median(values)) if values else np.nan


def _audio_features(
    samples: np.ndarray,
    words: int,
    word_times: Sequence[tuple[float, float]],
    segments: Sequence[dict[str, object]],
) -> dict[str, float]:
    import librosa
    import parselmouth
    from parselmouth.praat import call

    sample_rate = 16_000
    result = {name: np.nan for name in AUDIO_FEATURES}
    result.update(
        duration=len(samples) / sample_rate, speech_duration=np.nan, voiced_duration=np.nan
    )
    if len(samples) < sample_rate // 2 or not np.max(np.abs(samples)) > 1e-6:
        return result
    speech = _intervals_union(
        [(float(segment["start"]), float(segment["end"])) for segment in segments]
    )
    result["speech_duration"] = sum(end - start for start, end in speech)
    if not speech:
        return result
    span = max(speech[-1][1] - speech[0][0], 1e-6)
    timed = sorted((float(start), float(end)) for start, end in word_times if end > start)
    gaps = [max(0.0, timed[index + 1][0] - timed[index][1]) for index in range(len(timed) - 1)]
    gaps = [gap for gap in gaps if gap >= 0.20]
    result["pause_mean"] = _finite(np.mean(gaps)) if gaps else (0.0 if len(timed) > 1 else np.nan)
    result["pause_sd"] = _finite(np.std(gaps, ddof=1)) if len(gaps) > 1 else np.nan
    result["pause_fraction"] = _finite(min(1.0, sum(gaps) / span)) if len(timed) > 1 else np.nan
    result["speech_rate"] = _finite(60 * words / span) if words else np.nan
    timed_duration = sum(end - start for start, end in timed)
    result["articulation_rate"] = (
        _finite(60 * len(timed) / timed_duration) if timed_duration > 0.1 and timed else np.nan
    )

    sound = parselmouth.Sound(samples, sampling_frequency=sample_rate)
    pitch = sound.to_pitch(time_step=0.01, pitch_floor=65, pitch_ceiling=500)
    frequency = pitch.selected_array["frequency"]
    frequency = frequency[(frequency > 0) & np.isfinite(frequency)]
    result["voiced_duration"] = len(frequency) * 0.01
    if len(frequency) >= 30:
        result["pitch_sd"] = _finite(np.std(12 * np.log2(frequency / np.median(frequency)), ddof=1))
    harmonicity = sound.to_harmonicity_cc(time_step=0.01, minimum_pitch=65)
    result["hnr"] = _finite(call(harmonicity, "Get mean", 0, 0))
    intensity = sound.to_intensity(time_step=0.01, minimum_pitch=65)
    intensity_times = intensity.xs()
    intensity_values = intensity.values[0]
    speech_mask = np.zeros(len(intensity_times), dtype=bool)
    for start, end in speech:
        speech_mask |= (intensity_times >= start) & (intensity_times <= end)
    valid_intensity = intensity_values[speech_mask & np.isfinite(intensity_values)]
    if len(valid_intensity) >= 30:
        result["intensity_sd"] = _finite(np.std(valid_intensity, ddof=1))
    clips = [
        samples[
            max(0, int(start * sample_rate)) : min(
                len(samples), int(min(end, start + 15) * sample_rate)
            )
        ]
        for start, end in speech[:30]
    ]
    joined = np.concatenate(clips) if clips else np.array([], dtype=np.float32)
    if len(joined) >= sample_rate:
        result["cpp"] = _spectral_cpp(joined, sample_rate)
    point_process = call(sound, "To PointProcess (periodic, cc)", 65, 500)
    result["jitter"] = _finite(call(point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3))
    result["shimmer"] = _finite(
        call([sound, point_process], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    )

    mfcc = librosa.feature.mfcc(y=samples, sr=sample_rate, n_mfcc=13)
    for index in range(13):
        result[f"mfcc_{index + 1}"] = _finite(np.mean(mfcc[index]))
    power_spectrum = np.abs(librosa.stft(samples, n_fft=1024)) ** 2
    frequencies = librosa.fft_frequencies(sr=sample_rate, n_fft=1024)
    band = (frequencies >= 150) & (frequencies <= 4000)
    mean_power = np.mean(power_spectrum[band], axis=1)
    result["spectral_tilt"] = _finite(
        np.polyfit(np.log(frequencies[band]), 10 * np.log10(mean_power + 1e-12), 1)[0]
    )
    return result


class _TextModels:
    def __init__(self, syntax_models: Mapping[str, str] | None = None) -> None:
        self.embedder: object | None = None
        self.parsers: dict[str, object] = {}
        self.parser_names = {"en": "en_core_web_sm", "es": "es_core_news_sm"}
        self.parser_names.update(syntax_models or {})
        self.parser_load_attempted: set[str] = set()
        self.english_sentiment: object | None = None
        self.multilingual_sentiment: object | None = None
        from sentence_transformers import SentenceTransformer

        self.embedder = SentenceTransformer(
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )
        try:
            from textblob import TextBlob
        except ImportError:
            warnings.warn("TextBlob unavailable; English sentiment markers will be missing")
        else:
            self.english_sentiment = TextBlob

    def parser(self, language: str) -> object | None:
        if language in self.parsers:
            return self.parsers[language]
        model_name = self.parser_names.get(language)
        if model_name is None or language in self.parser_load_attempted:
            return None
        self.parser_load_attempted.add(language)
        try:
            import spacy
        except ImportError:
            warnings.warn(
                f"spaCy unavailable; syntax complexity will be missing for {language!r}"
            )
            return None
        try:
            parser = spacy.load(model_name)
        except OSError:
            warnings.warn(
                f"spaCy model {model_name!r} unavailable; syntax complexity "
                f"will be missing for {language!r}"
            )
            return None
        self.parsers[language] = parser
        return parser

    def sentiment(self, text: str, language: str) -> float:
        if language == "en" and self.english_sentiment is not None:
            return _finite(self.english_sentiment(text).sentiment.polarity)
        if self.multilingual_sentiment is None:
            from transformers import pipeline

            self.multilingual_sentiment = pipeline(
                "text-classification",
                model="lxyuan/distilbert-base-multilingual-cased-sentiments-student",
                truncation=True,
            )
        prediction = self.multilingual_sentiment(text[:2000], truncation=True)[0]
        sign = {"positive": 1.0, "negative": -1.0, "neutral": 0.0}.get(prediction["label"].lower())
        return _finite(sign * prediction["score"]) if sign is not None else np.nan


def _validate_text_patterns(
    text_patterns: Mapping[str, Mapping[str, object]] | None,
) -> dict[str, dict[str, object]]:
    allowed_sections = {"activities", "health", "temporal", "detail"}
    validated: dict[str, dict[str, object]] = {"en": dict(ENGLISH_PATTERNS)}
    for language, pattern_set in (text_patterns or {}).items():
        if not isinstance(language, str) or not language.strip():
            raise ValueError("text_patterns language keys must be non-empty language codes")
        if not isinstance(pattern_set, Mapping):
            raise TypeError(f"text_patterns[{language!r}] must be a mapping")
        unknown = set(pattern_set) - allowed_sections
        if unknown:
            raise ValueError(f"Unknown text pattern sections for {language!r}: {sorted(unknown)}")
        merged = dict(validated.get(language, {}))
        for section, patterns in pattern_set.items():
            if section in {"activities", "health"}:
                if not isinstance(patterns, Mapping):
                    raise ValueError(f"text_patterns[{language!r}][{section!r}] must be a mapping")
                expected = (
                    set(ACTIVITY_CATEGORIES) if section == "activities" else set(HEALTH_FEATURES)
                )
                unknown_patterns = set(patterns) - expected
                if unknown_patterns:
                    raise ValueError(
                        f"Unknown {section} pattern names for {language!r}: "
                        f"{sorted(unknown_patterns)}"
                    )
                existing = merged.get(section, {})
                merged[section] = (
                    {**existing, **patterns} if isinstance(existing, Mapping) else dict(patterns)
                )
                for expression in patterns.values():
                    if not isinstance(expression, str):
                        raise TypeError("text patterns must be regular-expression strings")
                    try:
                        re.compile(expression, re.IGNORECASE)
                    except re.error as exc:
                        raise ValueError(f"Invalid text pattern {expression!r}: {exc}") from exc
            else:
                if not isinstance(patterns, str):
                    raise TypeError(f"text_patterns[{language!r}][{section!r}] must be a string")
                try:
                    re.compile(patterns, re.IGNORECASE)
                except re.error as exc:
                    raise ValueError(f"Invalid text pattern {patterns!r}: {exc}") from exc
                merged[section] = patterns
        validated[language] = merged
    return validated


def _text_features(
    text: str,
    transcript_tokens: Sequence[str],
    utterances: Sequence[str],
    models: _TextModels,
    *,
    language: str,
    min_words: int,
    include_behavior: bool,
    include_health: bool,
    patterns: Mapping[str, object] | None,
) -> dict[str, float | int]:
    features: dict[str, float | int] = {name: np.nan for name in LANGUAGE_FEATURES}
    if include_behavior:
        features.update({name: np.nan for name in BEHAVIOR_FEATURES})
    if include_health:
        features.update({name: np.nan for name in HEALTH_FEATURES})
    words = [token.casefold() for token in transcript_tokens if token and not token.isspace()]
    if not words:
        words = [token.casefold() for token in WORD_PATTERN.findall(text)]
    count = len(words)
    informative = count >= min_words
    features.update(words=count, informative=int(informative))
    if count >= 30:
        features["lexical_diversity"] = _mtld(words)
    if count >= 8:
        try:
            from wordfreq import zipf_frequency
        except ImportError:
            warnings.warn("wordfreq unavailable; lexical-sophistication markers will be missing")
        else:
            parser = models.parser(language)
            stop_words = parser.Defaults.stop_words if parser is not None else set()
            content_words = [word for word in words if word not in stop_words and len(word) > 2]
            if len(content_words) >= 4:
                frequencies = [zipf_frequency(word, language) for word in content_words]
                features["lexical_sophistication"] = _finite(
                    np.mean([7 - min(7, max(0, frequency)) for frequency in frequencies])
                )
    sentences = [sentence.strip() for sentence in SENTENCE_PATTERN.split(text) if sentence.strip()]
    parser = models.parser(language) if count >= 6 else None
    if parser is not None and count >= 6:
        doc = parser(text[: parser.max_length])
        parsed_sentences = [sentence for sentence in doc.sents if sentence.text.strip()]
        clauses = sum(
            token.dep_ in {"ROOT", "ccomp", "xcomp", "advcl", "relcl", "acl", "conj"}
            and token.pos_ in {"VERB", "AUX"}
            for token in doc
        )
        if parsed_sentences:
            sentences = [sentence.text.strip() for sentence in parsed_sentences]
            features["syntax_complexity"] = _finite(clauses / len(parsed_sentences))
    if informative and models.embedder is not None:
        eligible = [
            utterance.strip()
            for utterance in (utterances or sentences)
            if utterance.strip() and len(WORD_PATTERN.findall(utterance)) >= 3
        ]
        if len(eligible) >= 2:
            vectors = models.embedder.encode(
                eligible, normalize_embeddings=True, show_progress_bar=False
            )
            similarities = np.sum(vectors[:-1] * vectors[1:], axis=1)
            features["topic_coherence"] = _finite(np.mean(similarities))
            features["topic_shift"] = _finite(np.mean(similarities < 0.25))
            features["semantic_repetition"] = _finite(np.mean(similarities > 0.90))
    if informative:
        if count >= 4:
            features["sentiment"] = models.sentiment(text, language)
        active_patterns = patterns or {}
        activity_patterns = active_patterns.get("activities", {})
        health_patterns = active_patterns.get("health", {})
        temporal_pattern = active_patterns.get("temporal")
        detail_pattern = active_patterns.get("detail")
        if isinstance(activity_patterns, Mapping):
            activity_hits = {
                name: bool(re.search(pattern, text, re.IGNORECASE))
                for name, pattern in activity_patterns.items()
            }
            if include_behavior and activity_patterns:
                features["activity_count"] = sum(
                    len(re.findall(pattern, text, re.IGNORECASE))
                    for pattern in activity_patterns.values()
                )
                features["activity_diversity"] = sum(activity_hits.values())
                for domain, categories in ACTIVITY_DOMAINS.items():
                    matching = [activity_hits[key] for key in categories if key in activity_hits]
                    if matching:
                        features[domain] = int(any(matching))
            if activity_patterns and (temporal_pattern or detail_pattern):
                matched_sentences = [
                    sentence
                    for sentence in sentences
                    if (language != "en" or len(WORD_PATTERN.findall(sentence)) >= 3)
                    if any(
                        re.search(pattern, sentence, re.IGNORECASE)
                        for pattern in activity_patterns.values()
                    )
                ]
                if matched_sentences:
                    features["specificity"] = _finite(
                        np.mean(
                            [
                                bool(
                                    (
                                        temporal_pattern
                                        and re.search(temporal_pattern, sentence, re.IGNORECASE)
                                    )
                                    or (
                                        detail_pattern
                                        and re.search(detail_pattern, sentence, re.IGNORECASE)
                                    )
                                )
                                for sentence in matched_sentences
                            ]
                        )
                    )
        if include_health and isinstance(health_patterns, Mapping):
            for marker, pattern in health_patterns.items():
                features[marker] = int(bool(re.search(pattern, text, re.IGNORECASE)))
    return features


def _mtld(tokens: list[str], threshold: float = 0.72) -> float:
    def direction(sequence: list[str]) -> float:
        seen: set[str] = set()
        count = 0
        factors = 0.0
        for token in sequence:
            seen.add(token)
            count += 1
            if len(seen) / count <= threshold:
                factors += 1
                seen = set()
                count = 0
        if count:
            factors += (1 - len(seen) / count) / (1 - threshold)
        return len(sequence) / factors if factors else np.nan

    return _finite(np.nanmean([direction(tokens), direction(tokens[::-1])]))


def _create_whisper_model(model_name: str, device: str) -> tuple[object, str]:
    import ctranslate2
    from faster_whisper import WhisperModel

    selected_device = (
        ("cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu")
        if device == "auto"
        else device
    )
    model = WhisperModel(
        model_name,
        device=selected_device,
        compute_type="float16" if selected_device == "cuda" else "int8",
    )
    return model, selected_device


def _analyze_recording(
    recording: AudioRecording,
    model: object,
    text_models: _TextModels,
    *,
    language: str,
    beam_size: int,
    min_words: int,
    include_behavior: bool,
    include_health: bool,
    pattern_sets: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    samples = _audio_samples(recording)
    source: str | np.ndarray = str(recording.path) if recording.path is not None else samples
    asr = _asr_features(source, model, language=language, beam_size=beam_size)
    detected_language = str(asr["language"])
    text = _text_features(
        str(asr["text"]),
        asr["tokens"],
        [str(segment["text"]) for segment in asr["segments"]],
        text_models,
        language=detected_language,
        min_words=min_words,
        include_behavior=include_behavior,
        include_health=include_health,
        patterns=pattern_sets.get(
            detected_language,
            pattern_sets.get(detected_language.split("-")[0]),
        ),
    )
    acoustics = _audio_features(
        samples,
        len(asr["tokens"]) or int(text["words"]),
        asr["word_times"],
        asr["segments"],
    )
    return {
        "filename": recording.filename,
        "date": _date_from_filename(recording.filename),
        **asr,
        **text,
        **acoustics,
    }


def _aggregate_recordings(recordings: pd.DataFrame, features: Sequence[str]) -> pd.DataFrame:
    languages = sorted(set(recordings["language"].dropna().astype(str)))
    result: dict[str, object] = {
        "id": recordings["id"].iloc[0],
        "n_recordings": len(recordings),
        "language": languages[0] if len(languages) == 1 else ("mixed" if languages else ""),
        "informative_count": int(recordings["informative"].fillna(0).sum()),
    }
    dated = recordings[recordings["date"].notna()]
    for feature in features:
        valid = pd.to_numeric(recordings[feature], errors="coerce").dropna()
        result[f"{feature}_mean"] = _finite(valid.mean()) if len(valid) else np.nan
        result[f"{feature}_n_valid"] = len(valid)
        if len(dated):
            by_date = dated.groupby("date")[feature].mean().dropna().sort_index()
            values = (
                by_date.to_numpy(dtype=float)
                if len(dated) == len(recordings)
                else valid.to_numpy(dtype=float)
            )
        else:
            values = valid.to_numpy(dtype=float)
        result[f"{feature}_sd"] = _finite(np.std(values, ddof=1)) if len(values) >= 2 else np.nan
        result[f"{feature}_slope"] = np.nan
        result[f"{feature}_slope_span"] = np.nan
        by_date = dated.groupby("date")[feature].mean().dropna().sort_index()
        if len(by_date) >= 2:
            days = (
                pd.to_datetime(by_date.index) - pd.to_datetime(by_date.index[0])
            ).days.to_numpy()
            years = days / 365.25
            if np.ptp(years) > 0:
                result[f"{feature}_slope"] = _finite(np.polyfit(years, by_date.to_numpy(), 1)[0])
                result[f"{feature}_slope_span"] = _finite(by_date.iloc[-1] - by_date.iloc[0])
    return pd.DataFrame([result])


def compute_audio_markers(
    data: ParticipantData,
    *,
    language: str = "auto",
    model: str = "large-v3",
    device: str = "auto",
    beam_size: int = 5,
    min_words: int = 8,
    include_behavior: bool = False,
    include_health: bool = False,
    text_patterns: Mapping[str, Mapping[str, object]] | None = None,
    syntax_models: Mapping[str, str] | None = None,
) -> MarkerResult:
    """Compute audio and transcript markers for a participant's recordings.

    Audio inputs are supplied in ``ParticipantData.audio`` as ``AudioRecording``
    instances. The returned per-recording table includes transcript text; nothing
    is written to disk by this function. English behavior and health patterns are
    built in; provide per-language ``text_patterns`` for other languages.
    """
    if not data.audio:
        raise ValueError("Missing required data for audio: audio")
    if not isinstance(language, str) or not language.strip():
        raise ValueError("language must be 'auto' or a non-empty language code")
    if device not in {"auto", "cpu", "cuda"}:
        raise ValueError("device must be 'auto', 'cpu', or 'cuda'")
    if beam_size < 1 or min_words < 1:
        raise ValueError("beam_size and min_words must be positive")

    pattern_sets = _validate_text_patterns(text_patterns)
    whisper_model, selected_device = _create_whisper_model(model, device)
    text_models = _TextModels(syntax_models)
    feature_names = list(AUDIO_FEATURES + LANGUAGE_FEATURES)
    if include_behavior:
        feature_names.extend(BEHAVIOR_FEATURES)
    if include_health:
        feature_names.extend(HEALTH_FEATURES)
    rows = []
    for recording in data.audio:
        row = _analyze_recording(
            recording,
            whisper_model,
            text_models,
            language=language,
            beam_size=beam_size,
            min_words=min_words,
            include_behavior=include_behavior,
            include_health=include_health,
            pattern_sets=pattern_sets,
        )
        row["id"] = data.participant_id
        rows.append(row)
    per_recording = pd.DataFrame(rows)
    for feature in feature_names:
        if feature not in per_recording:
            per_recording[feature] = np.nan
    markers = _aggregate_recordings(per_recording, feature_names)
    return MarkerResult(
        markers=markers,
        details={"recordings": per_recording},
        metadata={
            "marker": "audio",
            "participant_id": data.participant_id,
            "model": model,
            "device": selected_device,
            "language": language,
            "beam_size": beam_size,
            "min_words": min_words,
            "include_behavior": include_behavior,
            "include_health": include_health,
            "text_pattern_languages": sorted(text_patterns or {}),
            "syntax_model_languages": sorted(syntax_models or {}),
        },
        context=MarkerContext(data),
    )


def write_audio_transcripts(
    result: MarkerResult, output_directory: str | Path, *, suffix: str = "-transcript"
) -> list[Path]:
    """Write transcripts using each original filename stem and the optional suffix."""
    if "recordings" not in result.details:
        raise ValueError("MarkerResult does not contain details['recordings']")
    recordings = result.details["recordings"]
    required = {"filename", "text"}
    if not required.issubset(recordings.columns):
        raise ValueError("details['recordings'] must contain filename and text columns")

    targets: list[tuple[Path, str]] = []
    used_names: set[str] = set()
    for row in recordings.itertuples(index=False):
        filename = str(row.filename)
        text = row.text
        if not filename or filename == "nan" or not isinstance(text, str):
            raise ValueError("each recording must have a filename and transcript text")
        target_name = f"{Path(filename).stem}{suffix}.txt"
        if target_name.casefold() in used_names:
            raise ValueError(f"Transcript output name collision: {target_name}")
        used_names.add(target_name.casefold())
        targets.append((Path(output_directory) / target_name, text))

    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    written = []
    for target, text in targets:
        target.write_text(text, encoding="utf-8")
        written.append(target)
    return written


__all__ = ["compute_audio_markers", "write_audio_transcripts"]
