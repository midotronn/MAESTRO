"""Rank synchronized comparison windows using video motion and audio timing proxies.

This is a screening tool, not a replacement for visual review. It measures optical-flow timing,
beat alignment, onset fit, stalls, and motion continuity in each 480-pixel source lane, then ranks
windows by MAESTRO's margin over the stronger baseline.

Requires the project dependencies plus ``opencv-python-headless``.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from pathlib import Path

import cv2
import imageio_ffmpeg
import librosa
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks


METHODS = ("LODGE", "EDGE", "MAESTRO")
LANE_WIDTH = 480
CROP_TOP = 40
CROP_BOTTOM = 440
ANALYSIS_FPS = 15.0


def _audio_features(video: Path, frame_count: int) -> tuple[np.ndarray, np.ndarray]:
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    with tempfile.TemporaryDirectory() as tmp:
        wav = Path(tmp) / "audio.wav"
        subprocess.run(
            [
                ffmpeg,
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                str(video),
                "-vn",
                "-ac",
                "1",
                "-ar",
                "22050",
                "-c:a",
                "pcm_s16le",
                "-y",
                str(wav),
            ],
            check=True,
        )
        audio, sample_rate = librosa.load(wav, sr=22050, mono=True)

    hop_length = round(sample_rate / ANALYSIS_FPS)
    onset = librosa.onset.onset_strength(
        y=audio,
        sr=sample_rate,
        hop_length=hop_length,
        aggregate=np.median,
    )
    onset = np.asarray(onset[:frame_count], dtype=np.float64)
    if len(onset) < frame_count:
        onset = np.pad(onset, (0, frame_count - len(onset)))
    onset = onset / (np.percentile(onset, 95) + 1e-8)

    _, beat_frames = librosa.beat.beat_track(
        onset_envelope=onset,
        sr=sample_rate,
        hop_length=hop_length,
        units="frames",
    )
    beat_mask = np.zeros(frame_count, dtype=np.float64)
    beat_frames = np.asarray(beat_frames, dtype=int)
    beat_frames = beat_frames[(beat_frames >= 0) & (beat_frames < frame_count)]
    beat_mask[beat_frames] = 1.0
    return onset, beat_mask


def _motion_features(video: Path) -> tuple[dict[str, np.ndarray], float]:
    cap = cv2.VideoCapture(str(video))
    source_fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    duration = float(cap.get(cv2.CAP_PROP_FRAME_COUNT)) / source_fps
    frame_step = max(1, round(source_fps / ANALYSIS_FPS))

    previous: list[np.ndarray | None] = [None, None, None]
    motion: dict[str, list[float]] = {method: [] for method in METHODS}
    coverage: dict[str, list[float]] = {method: [] for method in METHODS}

    frame_index = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_index % frame_step:
            frame_index += 1
            continue
        frame_index += 1

        for lane, method in enumerate(METHODS):
            image = frame[CROP_TOP:CROP_BOTTOM, lane * LANE_WIDTH:(lane + 1) * LANE_WIDTH]
            image = cv2.resize(image, (120, 100), interpolation=cv2.INTER_AREA)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (5, 5), 0)
            if previous[lane] is None:
                motion[method].append(0.0)
                coverage[method].append(0.0)
            else:
                flow = cv2.calcOpticalFlowFarneback(
                    previous[lane],
                    gray,
                    None,
                    pyr_scale=0.5,
                    levels=2,
                    winsize=13,
                    iterations=2,
                    poly_n=5,
                    poly_sigma=1.1,
                    flags=0,
                )
                magnitude = np.linalg.norm(flow, axis=2)
                motion[method].append(float(np.percentile(magnitude, 90)))
                coverage[method].append(float(np.mean(magnitude > 0.12)))
            previous[lane] = gray

    cap.release()
    features: dict[str, np.ndarray] = {}
    for method in METHODS:
        features[f"{method}.motion"] = gaussian_filter1d(
            np.asarray(motion[method], dtype=np.float64), sigma=1.0
        )
        features[f"{method}.coverage"] = gaussian_filter1d(
            np.asarray(coverage[method], dtype=np.float64), sigma=1.0
        )
    return features, duration


def _best_lagged_correlation(a: np.ndarray, b: np.ndarray, max_lag: int = 3) -> float:
    if len(a) < 4 or np.std(a) < 1e-8 or np.std(b) < 1e-8:
        return 0.0
    correlations = []
    for lag in range(-max_lag, max_lag + 1):
        if lag < 0:
            left, right = a[-lag:], b[:lag]
        elif lag > 0:
            left, right = a[:-lag], b[lag:]
        else:
            left, right = a, b
        if len(left) >= 4:
            correlations.append(float(np.corrcoef(left, right)[0, 1]))
    return max(correlations, default=0.0)


def _beat_alignment(motion: np.ndarray, beat_mask: np.ndarray) -> float:
    beats = np.flatnonzero(beat_mask > 0)
    if not len(beats):
        return 0.0
    prominence = max(float(np.std(motion)) * 0.15, 1e-4)
    minima, _ = find_peaks(
        -motion,
        distance=max(2, round(ANALYSIS_FPS * 0.22)),
        prominence=prominence,
    )
    if not len(minima):
        return 0.0
    nearest = np.min(np.abs(beats[:, None] - minima[None, :]), axis=1) / ANALYSIS_FPS
    return float(np.mean(np.exp(-0.5 * (nearest / 0.14) ** 2)))


def _window_metrics(
    motion: np.ndarray,
    coverage: np.ndarray,
    onset: np.ndarray,
    beat_mask: np.ndarray,
) -> dict[str, float]:
    scale = float(np.percentile(motion, 75) + 1e-8)
    normalized = motion / scale
    acceleration = np.abs(np.diff(normalized, prepend=normalized[0]))
    jerk = np.abs(np.diff(normalized, n=2, prepend=[normalized[0], normalized[0]]))
    onset_fit = _best_lagged_correlation(onset, acceleration)
    beat_alignment = _beat_alignment(normalized, beat_mask)
    freeze_ratio = float(np.mean(normalized < 0.12))
    spike_ratio = float(np.percentile(normalized, 95) / (np.median(normalized) + 1e-8))
    jerk_mean = float(np.mean(jerk))
    coverage_mean = float(np.mean(coverage))

    quality = (
        0.42 * beat_alignment
        + 0.24 * max(0.0, onset_fit)
        + 0.18 * (1.0 - min(freeze_ratio, 1.0))
        + 0.10 * min(coverage_mean / 0.18, 1.0)
        + 0.06 * (1.0 / (1.0 + jerk_mean))
    )
    return {
        "quality": quality,
        "beat_alignment": beat_alignment,
        "onset_fit": onset_fit,
        "freeze_ratio": freeze_ratio,
        "jerk": jerk_mean,
        "spike_ratio": spike_ratio,
        "coverage": coverage_mean,
        "motion": float(np.mean(normalized)),
    }


def analyze_video(
    video: Path,
    window_seconds: float,
    stride_seconds: float,
    edge_seconds: float,
) -> dict[str, object]:
    features, duration = _motion_features(video)
    frame_count = min(len(features[f"{method}.motion"]) for method in METHODS)
    onset, beat_mask = _audio_features(video, frame_count)
    window_frames = round(window_seconds * ANALYSIS_FPS)
    stride_frames = round(stride_seconds * ANALYSIS_FPS)
    edge_frames = round(edge_seconds * ANALYSIS_FPS)

    rows: list[dict[str, object]] = []
    for start in range(edge_frames, frame_count - edge_frames - window_frames + 1, stride_frames):
        stop = start + window_frames
        method_metrics = {}
        for method in METHODS:
            method_metrics[method] = _window_metrics(
                features[f"{method}.motion"][start:stop],
                features[f"{method}.coverage"][start:stop],
                onset[start:stop],
                beat_mask[start:stop],
            )

        baseline_quality = max(
            method_metrics["LODGE"]["quality"],
            method_metrics["EDGE"]["quality"],
        )
        quality_margin = method_metrics["MAESTRO"]["quality"] - baseline_quality
        beat_margin = method_metrics["MAESTRO"]["beat_alignment"] - max(
            method_metrics["LODGE"]["beat_alignment"],
            method_metrics["EDGE"]["beat_alignment"],
        )
        freeze_margin = min(
            method_metrics["LODGE"]["freeze_ratio"],
            method_metrics["EDGE"]["freeze_ratio"],
        ) - method_metrics["MAESTRO"]["freeze_ratio"]
        jerk_margin = min(
            method_metrics["LODGE"]["jerk"],
            method_metrics["EDGE"]["jerk"],
        ) - method_metrics["MAESTRO"]["jerk"]
        rows.append(
            {
                "start_seconds": round(start / ANALYSIS_FPS, 3),
                "end_seconds": round(stop / ANALYSIS_FPS, 3),
                "quality_margin": quality_margin,
                "beat_margin": beat_margin,
                "freeze_margin": freeze_margin,
                "jerk_margin": jerk_margin,
                "methods": method_metrics,
            }
        )

    rows.sort(key=lambda row: float(row["quality_margin"]), reverse=True)
    return {
        "source": video.name,
        "duration_seconds": duration,
        "analysis_fps": ANALYSIS_FPS,
        "window_seconds": window_seconds,
        "stride_seconds": stride_seconds,
        "windows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("video_dir", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--window-seconds", type=float, default=10.0)
    parser.add_argument("--stride-seconds", type=float, default=1.0)
    parser.add_argument("--edge-seconds", type=float, default=2.0)
    args = parser.parse_args()

    reports = []
    for video in sorted(args.video_dir.glob("story_*_3way.mp4")):
        print(f"analyzing {video.name}", flush=True)
        reports.append(
            analyze_video(
                video,
                window_seconds=args.window_seconds,
                stride_seconds=args.stride_seconds,
                edge_seconds=args.edge_seconds,
            )
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"videos": reports}, indent=2), encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
