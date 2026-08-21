"""Audio preprocessing for Lodge (librosa) and EDGE (Jukebox)."""

from __future__ import annotations

import gc
import logging
import os
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

from agentlodge.config import FPS, Settings
from agentlodge.env_paths import lodge_import_paths, use_code_paths
from agentlodge.subprocess_runner import (
    run_edge_inference_subprocess,
    run_jukebox_extraction,
    run_lodge_inference_subprocess,
)

logger = logging.getLogger(__name__)

HOP_LENGTH = 512
SAMPLE_RATE = FPS * HOP_LENGTH
AUDIO_TIMING_CONTRACT_VERSION = "audio-timing-v1"


@dataclass
class SongMetadata:
    duration_seconds: float
    bpm: float
    beat_frames: np.ndarray
    wav_path: Path


@dataclass
class AudioDescriptor:
    """Human-readable acoustic summary used to drive costume generation."""

    duration_seconds: float
    bpm: float
    tempo_feel: str
    rms_energy: float
    energy_level: str
    spectral_centroid_hz: float
    brightness: str
    onset_rate_per_second: float
    rhythmic_density: str
    key: str
    mode: str
    mood: str


@dataclass
class PreprocessedAudio:
    wav_path: Path
    metadata: SongMetadata
    lodge_features: np.ndarray
    edge_feature_slices: list[np.ndarray] | None = None
    audio_descriptor: AudioDescriptor | None = None


def ensure_wav(audio_path: str | Path) -> Path:
    """Convert mp3/other formats to wav at the pipeline sample rate."""
    audio_path = Path(audio_path).resolve()
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    if audio_path.suffix.lower() == ".wav":
        return audio_path

    try:
        from pydub import AudioSegment
    except ImportError as exc:
        raise ImportError("pydub is required to convert non-wav audio") from exc

    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()
    out_path = Path(tmp.name)
    segment = AudioSegment.from_file(str(audio_path))
    segment = segment.set_frame_rate(SAMPLE_RATE).set_channels(1)
    segment.export(str(out_path), format="wav")
    return out_path


def _metadata_from_waveform(y: np.ndarray, sr: int, wav_path: Path) -> SongMetadata:
    duration = len(y) / sr
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr, hop_length=HOP_LENGTH)
    bpm = float(np.atleast_1d(tempo)[0])
    return SongMetadata(
        duration_seconds=float(duration),
        bpm=bpm,
        beat_frames=np.asarray(beat_frames, dtype=np.int64),
        wav_path=wav_path,
    )


def extract_song_metadata(wav_path: Path) -> SongMetadata:
    y, sr = librosa.load(str(wav_path), sr=SAMPLE_RATE)
    return _metadata_from_waveform(y, sr, wav_path)


def extract_editor_beat_artifacts(
    wav_path: Path,
    *,
    librosa_module=librosa,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the exact 30 FPS beat positions and normalized onset strengths."""
    y, sr = librosa_module.load(
        str(wav_path),
        sr=22_050,
        mono=True,
    )
    _, beat_frames = librosa_module.beat.beat_track(
        y=y,
        sr=sr,
        hop_length=HOP_LENGTH,
        units="frames",
    )
    beat_frames = np.asarray(beat_frames, dtype=np.int64).reshape(-1)
    beat_times = librosa_module.frames_to_time(
        beat_frames,
        sr=sr,
        hop_length=HOP_LENGTH,
    ) * FPS
    onset = librosa_module.onset.onset_strength(
        y=y,
        sr=sr,
        hop_length=HOP_LENGTH,
    )
    strengths = np.asarray(
        [
            np.max(onset[max(0, frame - 1) : min(len(onset), frame + 2)])
            if len(onset)
            else 0.0
            for frame in beat_frames
        ],
        dtype=np.float32,
    )
    strengths = np.nan_to_num(
        strengths,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    if strengths.size and float(strengths.max()) > 0.0:
        strengths /= float(strengths.max())
    return np.asarray(beat_times, dtype=np.float32), strengths


_PITCH_CLASSES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
# Krumhansl-Schmuckler major/minor key profiles.
_KS_MAJOR = np.array(
    [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
)
_KS_MINOR = np.array(
    [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]
)


def _estimate_key_mode(chroma_mean: np.ndarray) -> tuple[str, str]:
    """Estimate musical key and major/minor mode via KS profile correlation."""
    if chroma_mean.size != 12 or not np.any(chroma_mean):
        return "C", "major"

    def _best(profile: np.ndarray) -> tuple[int, float]:
        best_idx, best_corr = 0, -np.inf
        for i in range(12):
            rolled = np.roll(chroma_mean, -i)
            if np.std(rolled) == 0:
                continue
            corr = float(np.corrcoef(rolled, profile)[0, 1])
            if np.isfinite(corr) and corr > best_corr:
                best_idx, best_corr = i, corr
        return best_idx, best_corr

    major_idx, major_corr = _best(_KS_MAJOR)
    minor_idx, minor_corr = _best(_KS_MINOR)
    if minor_corr > major_corr:
        return _PITCH_CLASSES[minor_idx], "minor"
    return _PITCH_CLASSES[major_idx], "major"


def _tempo_feel(bpm: float) -> str:
    if bpm < 80:
        return "slow"
    if bpm < 120:
        return "moderate"
    if bpm < 150:
        return "upbeat"
    return "fast"


def _energy_level(rms: float) -> str:
    if rms < 0.03:
        return "gentle"
    if rms < 0.08:
        return "moderate"
    return "energetic"


def _brightness(centroid_hz: float) -> str:
    if centroid_hz < 1500:
        return "dark and warm"
    if centroid_hz < 3000:
        return "balanced"
    return "bright and airy"


def _rhythmic_density(onset_rate: float) -> str:
    if onset_rate < 2.0:
        return "sparse"
    if onset_rate < 4.0:
        return "moderate"
    return "busy"


def extract_audio_descriptor(
    y: np.ndarray, sr: int, metadata: SongMetadata
) -> AudioDescriptor:
    """Compute a compact, human-readable acoustic summary of the song."""
    rms = float(np.mean(librosa.feature.rms(y=y)))
    centroid = float(
        np.mean(
            librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=HOP_LENGTH)
        )
    )
    onsets = librosa.onset.onset_detect(
        y=y, sr=sr, hop_length=HOP_LENGTH, units="frames"
    )
    onset_rate = (
        float(len(onsets) / metadata.duration_seconds)
        if metadata.duration_seconds
        else 0.0
    )
    chroma_mean = librosa.feature.chroma_stft(
        y=y, sr=sr, hop_length=HOP_LENGTH
    ).mean(axis=1)
    key, mode = _estimate_key_mode(chroma_mean)
    mood = "uplifting and bright" if mode == "major" else "moody and introspective"
    return AudioDescriptor(
        duration_seconds=metadata.duration_seconds,
        bpm=metadata.bpm,
        tempo_feel=_tempo_feel(metadata.bpm),
        rms_energy=rms,
        energy_level=_energy_level(rms),
        spectral_centroid_hz=centroid,
        brightness=_brightness(centroid),
        onset_rate_per_second=onset_rate,
        rhythmic_density=_rhythmic_density(onset_rate),
        key=key,
        mode=mode,
        mood=mood,
    )


def extract_lodge_features(wav_path: Path, lodge_code_path: Path) -> np.ndarray:
    """Extract 35-dim librosa features at 30 FPS for the full song."""
    with use_code_paths(lodge_code_path):
        from dld.data.utils.audio import extract as extract_music35

    features, _ = extract_music35(str(wav_path))
    return np.asarray(features, dtype=np.float32)


def slice_audio(wav_path: Path, stride: float, length: float, out_dir: Path) -> int:
    audio, sr = librosa.load(str(wav_path), sr=None)
    file_name = wav_path.stem
    start_idx = 0
    idx = 0
    window = int(length * sr)
    stride_step = int(stride * sr)
    out_dir.mkdir(parents=True, exist_ok=True)
    while start_idx <= len(audio) - window:
        audio_slice = audio[start_idx : start_idx + window]
        sf.write(out_dir / f"{file_name}_slice{idx}.wav", audio_slice, sr)
        start_idx += stride_step
        idx += 1
    return idx


def extract_edge_slices(
    wav_path: Path,
    edge_code_path: Path,
    work_dir: Path,
    *,
    max_slices: int | None = None,
) -> list[np.ndarray]:
    """Slice audio and extract Jukebox embeddings via an isolated EDGE subprocess."""
    slice_dir = (work_dir / "edge_slices").resolve()
    count = slice_audio(wav_path.resolve(), stride=2.5, length=5.0, out_dir=slice_dir)
    logger.info("Created %d EDGE audio slices in %s", count, slice_dir)

    wav_slices = sorted(slice_dir.glob("*.wav"), key=_slice_key)
    if not wav_slices:
        raise ValueError("EDGE preprocessing produced no audio slices")

    if max_slices is not None and len(wav_slices) > max_slices:
        logger.warning(
            "Limiting EDGE slices from %d to %d to reduce memory use",
            len(wav_slices),
            max_slices,
        )
        extra = slice_dir / "unused_slices"
        extra.mkdir(exist_ok=True)
        for wav_slice in wav_slices[max_slices:]:
            wav_slice.rename(extra / wav_slice.name)
        wav_slices = wav_slices[:max_slices]

    cache_dir = work_dir / "edge_juke_cache"
    pending = [
        cache_dir / f"{wav_slice.stem}.npy"
        for wav_slice in wav_slices
        if not (cache_dir / f"{wav_slice.stem}.npy").exists()
    ]
    if pending:
        run_jukebox_extraction(
            edge_code_path.resolve(), slice_dir.resolve(), cache_dir.resolve()
        )

    features: list[np.ndarray] = []
    for wav_slice in wav_slices:
        cache_path = cache_dir / f"{wav_slice.stem}.npy"
        if not cache_path.exists():
            raise RuntimeError(f"Missing Jukebox cache for {wav_slice.name}")
        features.append(np.load(cache_path).astype(np.float32))
    return features


def release_torch_memory() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            torch.mps.empty_cache()
    except Exception:
        pass


def _slice_key(path: Path) -> int:
    stem = path.stem
    return int(stem.split("slice")[-1])


def preprocess_audio(
    audio_path: str | Path,
    settings: Settings,
    work_dir: Path,
    *,
    extract_lodge: bool = True,
    extract_edge: bool = True,
) -> PreprocessedAudio:
    wav_path = ensure_wav(audio_path)
    y, sr = librosa.load(str(wav_path), sr=SAMPLE_RATE)
    metadata = _metadata_from_waveform(y, sr, wav_path)
    audio_descriptor = extract_audio_descriptor(y, sr, metadata)

    lodge_features = np.empty((0, 35), dtype=np.float32)
    if extract_lodge:
        lodge_features = extract_lodge_features(wav_path, settings.lodge_code_path)

    edge_slices = None
    if extract_edge:
        max_slices = settings.max_edge_slices
        edge_slices = extract_edge_slices(
            wav_path, settings.edge_code_path, work_dir, max_slices=max_slices
        )

    return PreprocessedAudio(
        wav_path=wav_path,
        metadata=metadata,
        lodge_features=lodge_features,
        edge_feature_slices=edge_slices,
        audio_descriptor=audio_descriptor,
    )
