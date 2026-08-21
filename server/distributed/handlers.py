"""GPU task handlers used by persistent RunPod workers."""

from __future__ import annotations

import glob
import hashlib
import json
import logging
import os
import pickle
import shutil
import subprocess
import sys
import tempfile
import uuid
import wave
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np

from server.distributed.render_contract import (
    RENDER_CONTRACT_VERSION,
    RGB_DIGEST_VERSION,
    WORKER_SHARD_VALIDATION_VERSION,
    probe_ffv1_shard,
    render_identity_digest,
    source_sequence_rgb_sha256,
)

logger = logging.getLogger(__name__)


def _shared_path(
    value: Any,
    shared_root: Path,
    *,
    must_exist: bool = False,
    suffix: str | None = None,
) -> Path:
    path = Path(str(value)).resolve()
    root = shared_root.resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"path is outside the shared root: {path}")
    if suffix and path.suffix.lower() != suffix.lower():
        raise ValueError(f"expected a {suffix} path, got {path}")
    if must_exist and not path.is_file():
        raise FileNotFoundError(path)
    return path


class JukeboxExtractHandler:
    """Keep one Jukebox model resident and extract exact per-slice features."""

    def __init__(
        self,
        *,
        edge_root: Path,
        shared_root: Path,
        extractor: Callable[[str], tuple[Any, Any]] | None = None,
        preload_audio_seconds: float | None = None,
        force_empty_cache: bool | None = None,
    ):
        self.edge_root = Path(edge_root).resolve()
        self.shared_root = Path(shared_root).resolve()
        self._extractor = extractor
        self._custom_extractor = extractor is not None
        self._jukebox_lib = None
        self._jukebox_package = None
        self._layer = 66
        self._fps = 30
        configured_seconds = (
            os.environ.get("AGENTLODGE_JUKEBOX_WARMUP_SECONDS", "5")
            if preload_audio_seconds is None
            else preload_audio_seconds
        )
        self.preload_audio_seconds = max(0.0, float(configured_seconds))
        configured_empty_cache = (
            os.environ.get("AGENTLODGE_JUKEBOX_EMPTY_CACHE", "0")
            if force_empty_cache is None
            else force_empty_cache
        )
        self.force_empty_cache = str(configured_empty_cache).strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }

    def _load_extractor(self) -> Callable[[str], tuple[Any, Any]]:
        if self._extractor is not None:
            return self._extractor
        if not self.edge_root.is_dir():
            raise FileNotFoundError(f"EDGE root does not exist: {self.edge_root}")
        os.chdir(self.edge_root)
        if str(self.edge_root) not in sys.path:
            sys.path.insert(0, str(self.edge_root))
        from data.audio_extraction import jukebox_features
        import jukemirlib
        import jukemirlib.lib as jukebox_lib

        self._extractor = jukebox_features.extract
        self._jukebox_lib = jukebox_lib
        self._jukebox_package = jukemirlib
        self._layer = int(jukebox_features.LAYER)
        self._fps = int(jukebox_features.FPS)
        return self._extractor

    def _extract_representations(self, wav_path: Path) -> np.ndarray:
        extractor = self._load_extractor()
        if self._custom_extractor or self._jukebox_lib is None:
            representations, _ = extractor(str(wav_path))
            return np.asarray(representations)
        audio = self._jukebox_lib.load_audio(str(wav_path))
        activations = self._jukebox_lib.extract(
            audio=audio,
            layers=[self._layer],
            downsample_target_rate=self._fps,
            force_empty_cache=self.force_empty_cache,
        )
        return np.asarray(activations[self._layer])

    def preload(self) -> None:
        self._load_extractor()
        if self._jukebox_lib is None:
            import jukemirlib

            if jukemirlib.VQVAE is None or jukemirlib.TOP_PRIOR is None:
                jukemirlib.VQVAE, jukemirlib.TOP_PRIOR = (
                    jukemirlib.setup_models()
                )
        else:
            if (
                self._jukebox_lib.VQVAE is None
                or self._jukebox_lib.TOP_PRIOR is None
            ):
                (
                    self._jukebox_lib.VQVAE,
                    self._jukebox_lib.TOP_PRIOR,
                ) = self._jukebox_lib.setup_models()
            self._jukebox_package.VQVAE = self._jukebox_lib.VQVAE
            self._jukebox_package.TOP_PRIOR = self._jukebox_lib.TOP_PRIOR
        if self.preload_audio_seconds <= 0:
            return

        sample_rate = 44_100
        sample_count = max(1, round(sample_rate * self.preload_audio_seconds))
        time_axis = np.arange(sample_count, dtype=np.float64) / sample_rate
        pcm = np.asarray(
            np.sin(2.0 * np.pi * 440.0 * time_axis) * 3276,
            dtype="<i2",
        )
        previous_directory = Path.cwd()
        with tempfile.TemporaryDirectory(
            prefix="maestro-jukebox-warmup-"
        ) as temporary:
            temporary_path = Path(temporary)
            audio_path = temporary_path / "warmup.wav"
            with wave.open(str(audio_path), "wb") as audio:
                audio.setnchannels(1)
                audio.setsampwidth(2)
                audio.setframerate(sample_rate)
                audio.writeframes(pcm.tobytes())
            try:
                os.chdir(temporary_path)
                representations = self._extract_representations(audio_path)
            finally:
                os.chdir(previous_directory)
        warmed = np.asarray(representations)
        if warmed.size == 0 or not np.isfinite(warmed).all():
            raise RuntimeError("Jukebox preload inference produced invalid features")

    def __call__(self, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        items = payload.get("items")
        if not isinstance(items, list) or not items:
            raise ValueError("jukebox.extract requires a non-empty items list")
        self._load_extractor()
        outputs: list[str] = []
        cached = 0
        for item in items:
            if not isinstance(item, Mapping):
                raise ValueError("jukebox item must be an object")
            wav_path = _shared_path(
                item.get("wav"),
                self.shared_root,
                must_exist=True,
                suffix=".wav",
            )
            output_path = _shared_path(
                item.get("output"),
                self.shared_root,
                suffix=".npy",
            )
            output_path.parent.mkdir(parents=True, exist_ok=True)
            if output_path.is_file() and output_path.stat().st_size > 0:
                cached += 1
                outputs.append(str(output_path))
                continue
            representations = self._extract_representations(wav_path)
            temporary = output_path.with_name(
                output_path.name + f".{uuid.uuid4().hex}.tmp.npy"
            )
            try:
                np.save(
                    temporary,
                    np.asarray(representations, dtype=np.float32),
                )
                os.replace(temporary, output_path)
            finally:
                temporary.unlink(missing_ok=True)
            outputs.append(str(output_path))
        return {
            "items": len(items),
            "cached": cached,
            "outputs": outputs,
        }


def _atomic_save_array(path: Path, value: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{uuid.uuid4().hex}.tmp.npy")
    try:
        np.save(temporary, np.asarray(value, dtype=np.float32))
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _source_fingerprint(path: Path) -> dict[str, object]:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return {
        "bytes": path.stat().st_size,
        "sha256": digest.hexdigest(),
    }


class AudioPreprocessHandler:
    """Keep the librosa/Numba stack warm for role-specific audio features."""

    def __init__(
        self,
        *,
        mode: str,
        shared_root: Path,
        lodge_root: Path | None = None,
        edge_root: Path | None = None,
    ):
        if mode not in {"lodge", "edge"}:
            raise ValueError(f"unsupported audio preprocessing mode: {mode}")
        self.mode = mode
        self.shared_root = Path(shared_root).resolve()
        self.lodge_root = (
            Path(lodge_root).resolve() if lodge_root is not None else None
        )
        self.edge_root = (
            Path(edge_root).resolve() if edge_root is not None else None
        )

    def preload(self) -> None:
        from agentlodge.audio.preprocess import (
            extract_lodge_features,
            slice_audio,
        )

        sample_rate = 15_360
        sample_count = sample_rate * 5
        time_axis = np.arange(sample_count, dtype=np.float64) / sample_rate
        pcm = np.asarray(
            np.sin(2.0 * np.pi * 440.0 * time_axis) * 3276,
            dtype="<i2",
        )
        with tempfile.TemporaryDirectory(
            prefix=f"maestro-audio-{self.mode}-warmup-"
        ) as temporary:
            root = Path(temporary)
            audio_path = root / "warmup.wav"
            with wave.open(str(audio_path), "wb") as audio:
                audio.setnchannels(1)
                audio.setsampwidth(2)
                audio.setframerate(sample_rate)
                audio.writeframes(pcm.tobytes())
            if self.mode == "lodge":
                if self.lodge_root is None:
                    raise RuntimeError("LODGE audio worker has no code root")
                features = extract_lodge_features(
                    audio_path,
                    self.lodge_root,
                )
                if features.ndim != 2 or features.shape[1] != 35:
                    raise RuntimeError(
                        f"LODGE audio preload returned shape {features.shape}"
                    )
            else:
                count = slice_audio(
                    audio_path,
                    stride=2.5,
                    length=5.0,
                    out_dir=root / "slices",
                )
                if count != 1:
                    raise RuntimeError(
                        f"EDGE audio preload produced {count} slices"
                    )

    def __call__(self, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        from agentlodge.audio.preprocess import (
            extract_edge_slices,
            extract_lodge_features,
        )

        wav_path = _shared_path(
            payload.get("wav"),
            self.shared_root,
            must_exist=True,
            suffix=".wav",
        )
        output_path = _shared_path(
            payload.get("output"),
            self.shared_root,
            suffix=".npy",
        )
        if self.mode == "lodge":
            if self.lodge_root is None:
                raise RuntimeError("LODGE audio worker has no code root")
            value = extract_lodge_features(wav_path, self.lodge_root)
        else:
            if self.edge_root is None:
                raise RuntimeError("EDGE audio worker has no code root")
            work_dir = _shared_path(
                payload.get("work_dir"),
                self.shared_root,
            )
            value = np.asarray(
                extract_edge_slices(
                    wav_path,
                    self.edge_root,
                    work_dir,
                ),
                dtype=np.float32,
            )
        value = np.asarray(value, dtype=np.float32)
        _atomic_save_array(output_path, value)
        return {
            "output": str(output_path),
            "shape": [int(dimension) for dimension in value.shape],
            "dtype": str(value.dtype),
        }


class BeatArtifactHandler:
    """Keep both exact librosa beat-analysis paths compiled and resident."""

    def __init__(self, *, shared_root: Path):
        self.shared_root = Path(shared_root).resolve()

    @staticmethod
    def _analyze(wav_path: Path):
        from agentlodge.audio.preprocess import (
            extract_editor_beat_artifacts,
            extract_song_metadata,
        )

        metadata = extract_song_metadata(wav_path)
        beats, strengths = extract_editor_beat_artifacts(wav_path)
        return metadata, beats, strengths

    def preload(self) -> None:
        sample_rate = 22_050
        sample_count = sample_rate * 5
        time_axis = np.arange(sample_count, dtype=np.float64) / sample_rate
        pcm = np.asarray(
            np.sin(2.0 * np.pi * 440.0 * time_axis) * 3276,
            dtype="<i2",
        )
        with tempfile.TemporaryDirectory(
            prefix="maestro-audio-beats-warmup-"
        ) as temporary:
            wav_path = Path(temporary) / "warmup.wav"
            with wave.open(str(wav_path), "wb") as audio:
                audio.setnchannels(1)
                audio.setsampwidth(2)
                audio.setframerate(sample_rate)
                audio.writeframes(pcm.tobytes())
            metadata, beats, strengths = self._analyze(wav_path)
        if (
            metadata.duration_seconds <= 0.0
            or beats.ndim != 1
            or strengths.shape != beats.shape
            or not np.isfinite(beats).all()
            or not np.isfinite(strengths).all()
        ):
            raise RuntimeError("beat-analysis preload produced invalid artifacts")

    def __call__(self, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        from agentlodge.audio.preprocess import AUDIO_TIMING_CONTRACT_VERSION

        sid = str(payload.get("sid") or "").strip()
        if not sid or any(
            character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
            for character in sid
        ):
            raise ValueError(f"invalid song id: {sid!r}")
        wav_path = _shared_path(
            payload.get("wav"),
            self.shared_root,
            must_exist=True,
            suffix=".wav",
        )
        metadata_path = _shared_path(
            payload.get("metadata_output"),
            self.shared_root,
            suffix=".json",
        )
        beats_path = _shared_path(
            payload.get("beats_output"),
            self.shared_root,
            suffix=".npy",
        )
        strengths_path = _shared_path(
            payload.get("strengths_output"),
            self.shared_root,
            suffix=".npy",
        )
        metadata, beats, strengths = self._analyze(wav_path)
        _atomic_save_array(beats_path, beats)
        _atomic_save_array(strengths_path, strengths)
        _atomic_write_json(
            metadata_path,
            {
                "contract_version": AUDIO_TIMING_CONTRACT_VERSION,
                "source": _source_fingerprint(wav_path),
                "duration_seconds": float(metadata.duration_seconds),
                "bpm": float(metadata.bpm),
                "beat_frames": np.asarray(
                    metadata.beat_frames,
                    dtype=np.int64,
                ).reshape(-1).tolist(),
            },
        )
        return {
            "sid": sid,
            "metadata_output": str(metadata_path),
            "beats_output": str(beats_path),
            "strengths_output": str(strengths_path),
            "metadata_beats": int(np.asarray(metadata.beat_frames).size),
            "editor_beats": int(beats.size),
        }


PENETRATION_COMPLETION_MARKER = "penetration_cleanup_{sid}.done"
PENETRATION_RADIUS = 0.12
PENETRATION_MARGIN = 0.03
PENETRATION_MAX_DEG = 30.0


class DanceGenerationHandler:
    """Keep generation orchestration and motion-assembly imports resident."""

    def __init__(
        self,
        *,
        shared_root: Path,
        penetration_resolver: Any | None = None,
    ):
        self.shared_root = Path(shared_root).resolve()
        self._generate_song = None
        self._build_bank = None
        self._penetration_resolver = penetration_resolver

    def _generator(self):
        if self._generate_song is None:
            from scripts.make_song_bestofk import generate_song

            self._generate_song = generate_song
        return self._generate_song

    def _bank_builder(self):
        if self._build_bank is None:
            from scripts.build_window_bank import build_bank

            self._build_bank = build_bank
        return self._build_bank

    def _resolver(self):
        if self._penetration_resolver is None:
            from scripts.resolve_penetration import PenetrationResolver

            self._penetration_resolver = PenetrationResolver(
                workspace=self.shared_root,
            )
        return self._penetration_resolver

    def _penetration_marker(self, sid: str) -> Path:
        return self.shared_root / PENETRATION_COMPLETION_MARKER.format(sid=sid)

    @staticmethod
    def _write_completion_marker(path: Path) -> None:
        temporary = path.with_name(path.name + f".{uuid.uuid4().hex}.tmp")
        try:
            temporary.write_text(
                f"radius={PENETRATION_RADIUS:.2f} "
                f"margin={PENETRATION_MARGIN:.2f} "
                f"max_deg={PENETRATION_MAX_DEG:g}\n",
                encoding="utf-8",
            )
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)

    def preload(self) -> None:
        self._generator()
        self._bank_builder()
        self._resolver().preload()

    def __call__(self, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        sid = str(payload.get("sid") or "").strip()
        if not sid or any(
            character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
            for character in sid
        ):
            raise ValueError(f"invalid song id: {sid!r}")
        operation = str(payload.get("operation") or "generate_song").strip()
        if operation == "build_bank":
            bank_k = int(payload.get("bank_k") or 1)
            if not 1 <= bank_k <= 16:
                raise ValueError(f"bank_k must be between 1 and 16, got {bank_k}")
            return self._bank_builder()(
                sid,
                bank_k,
                workspace=self.shared_root,
                distributed=True,
            )
        if operation != "generate_song":
            raise ValueError(f"unsupported dance generation operation: {operation!r}")
        cleanup_requested = payload.get("penetration_cleanup", False)
        if not isinstance(cleanup_requested, bool):
            raise ValueError("penetration_cleanup must be a boolean")
        for path in (
            self.shared_root / "LODGE/data/finedance/music_wav" / f"{sid}.wav",
            self.shared_root / f"lodge_fd_{sid}_feats.npy",
            self.shared_root / f"edge{sid}_slices.npy",
        ):
            if not path.is_file():
                raise FileNotFoundError(path)
        timing_value = str(payload.get("timing_file") or "").strip()
        timing_path = (
            _shared_path(timing_value, self.shared_root)
            if timing_value
            else None
        )
        previous_timing = os.environ.get("MAESTRO_TIMING_FILE")
        try:
            if timing_path is not None:
                timing_path.parent.mkdir(parents=True, exist_ok=True)
                os.environ["MAESTRO_TIMING_FILE"] = str(timing_path)
            report = self._generator()(sid)
            output = self.shared_root / f"fd_{sid}_STORY_bestofk.npy"
            if not output.is_file() or output.stat().st_size == 0:
                raise RuntimeError(f"generation did not produce {output}")
            if cleanup_requested:
                marker = self._penetration_marker(sid)
                marker.unlink(missing_ok=True)
                resolved = self._resolver().resolve(
                    np.load(output).astype(np.float32),
                    radius=PENETRATION_RADIUS,
                    margin=PENETRATION_MARGIN,
                    max_deg=PENETRATION_MAX_DEG,
                )
                _atomic_save_array(output, resolved)
                self._write_completion_marker(marker)
            self._bank_builder()(
                sid,
                1,
                workspace=self.shared_root,
                distributed=True,
            )
        finally:
            if previous_timing is None:
                os.environ.pop("MAESTRO_TIMING_FILE", None)
            else:
                os.environ["MAESTRO_TIMING_FILE"] = previous_timing
        return {
            "sid": sid,
            "output": str(output),
            "frames": int(report["frames"]),
            "best_of_k": int(report["best_of_k"]),
            "generation_workers": report.get("generation_workers") or {},
        }


def _ffmpeg_executable() -> str:
    configured = os.environ.get("AGENTLODGE_FFMPEG", "").strip()
    if configured:
        return configured
    executable = shutil.which("ffmpeg")
    if executable:
        return executable
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError as exc:
        raise FileNotFoundError(
            "ffmpeg is required for lossless render-shard packaging"
        ) from exc


def _frame_digest(
    frames_dir: Path,
    frame_start: int,
    frame_end: int,
    frame_format: str,
) -> str:
    digest = hashlib.sha256()
    for frame in range(frame_start, frame_end):
        path = frames_dir / f"frame_{frame:04d}.{frame_format}"
        digest.update(path.name.encode("ascii"))
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def _package_ffv1(
    frames_dir: Path,
    output_path: Path,
    *,
    frame_start: int,
    frame_end: int,
    frame_format: str,
    fps: int,
) -> None:
    frame_count = frame_end - frame_start
    pattern = frames_dir / f"frame_%04d.{frame_format}"
    timeout = max(180, int(frame_count * 0.3 + 60))
    result = subprocess.run(
        [
            _ffmpeg_executable(),
            "-y",
            "-loglevel",
            "error",
            "-framerate",
            str(fps),
            "-start_number",
            str(frame_start),
            "-i",
            str(pattern),
            "-frames:v",
            str(frame_count),
            "-vf",
            "format=rgb24",
            "-c:v",
            "ffv1",
            "-level",
            "3",
            "-coder",
            "1",
            "-context",
            "1",
            "-g",
            "1",
            "-slicecrc",
            "1",
            str(output_path),
        ],
        capture_output=True,
        timeout=timeout,
    )
    if (
        result.returncode != 0
        or not output_path.is_file()
        or output_path.stat().st_size == 0
    ):
        detail = result.stderr.decode("utf-8", errors="replace")[-800:]
        raise RuntimeError(f"FFV1 shard packaging failed: {detail}")


class LodgeGenerateHandler:
    """Serve full-song LODGE requests while retaining model caches in-process."""

    def __init__(
        self,
        *,
        shared_root: Path,
        lodge_root: Path,
        lodge_weights: Path,
        lodge_global_weights: Path,
        genre: str = "Hiphop",
    ):
        self.shared_root = Path(shared_root).resolve()
        self.lodge_root = Path(lodge_root).resolve()
        self.lodge_weights = Path(lodge_weights).resolve()
        self.lodge_global_weights = Path(lodge_global_weights).resolve()
        self.genre = genre

    def _settings(self):
        from agentlodge.config import Settings

        return Settings.from_dict(
            {
                "lodge_code_path": str(self.lodge_root),
                "lodge_weights_path": str(self.lodge_weights),
                "lodge_global_weights_path": str(
                    self.lodge_global_weights
                ),
                "edge_code_path": str(self.lodge_root),
                "edge_weights_path": str(self.lodge_weights),
                "lodge_genre": self.genre,
                "max_edge_slices": None,
            }
        )

    def preload(self) -> None:
        for path in (
            self.lodge_root,
            self.lodge_weights,
            self.lodge_global_weights,
        ):
            if not path.exists():
                raise FileNotFoundError(path)
        from agentlodge.dance.lodge import generate_lodge_dance

        generate_lodge_dance(
            np.zeros((1, 35), dtype=np.float32),
            self._settings(),
            Path(tempfile.gettempdir()) / "maestro-lodge-preload",
            preload_only=True,
        )

    def __call__(self, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        from agentlodge.dance.lodge import generate_lodge_dance

        features_path = _shared_path(
            payload.get("features"),
            self.shared_root,
            must_exist=True,
            suffix=".npy",
        )
        output_path = _shared_path(
            payload.get("output"),
            self.shared_root,
            suffix=".npy",
        )
        work_dir = _shared_path(payload.get("work_dir"), self.shared_root)
        seed_value = payload.get("seed")
        seed = None if seed_value is None else int(seed_value)
        features = np.load(features_path).astype(np.float32)
        result = generate_lodge_dance(
            features,
            self._settings(),
            work_dir,
            seed=seed,
        )
        _atomic_save_array(output_path, result.motion)
        return {
            "output": str(output_path),
            "frames": int(result.motion.shape[0]),
            "summary": result.summary,
            "seed": seed,
        }


class EdgeGenerateHandler:
    """Serve full-song EDGE requests with one checkpoint loaded per worker."""

    def __init__(
        self,
        *,
        shared_root: Path,
        edge_root: Path,
        checkpoint: Path,
    ):
        self.shared_root = Path(shared_root).resolve()
        self.edge_root = Path(edge_root).resolve()
        self.checkpoint = Path(checkpoint).resolve()
        self._model = None
        self._torch = None

    def _runtime(self):
        if not self.edge_root.is_dir():
            raise FileNotFoundError(self.edge_root)
        if not self.checkpoint.is_file():
            raise FileNotFoundError(self.checkpoint)
        os.chdir(self.edge_root)
        if str(self.edge_root) not in sys.path:
            sys.path.insert(0, str(self.edge_root))
        if self._torch is None:
            import torch

            self._torch = torch
        if self._model is None:
            from EDGE import EDGE

            self._model = EDGE("jukebox", str(self.checkpoint))
            self._model.eval()
        return self._torch, self._model

    def preload(self) -> None:
        self._runtime()

    def _pkl_to_edge151(self, path: Path) -> np.ndarray:
        torch, _ = self._runtime()
        from dataset.quaternion import ax_to_6v

        with path.open("rb") as handle:
            data = pickle.load(handle)
        translation = data["smpl_trans"].astype(np.float32)
        poses = data["smpl_poses"].reshape(-1, 24, 3)
        rotations = (
            ax_to_6v(torch.from_numpy(poses))
            .numpy()
            .reshape(len(translation), 144)
        )
        contact = np.zeros((len(translation), 4), dtype=np.float32)
        return np.concatenate([translation, rotations, contact], axis=1)

    def __call__(self, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        features_path = _shared_path(
            payload.get("features"),
            self.shared_root,
            must_exist=True,
            suffix=".npy",
        )
        output_path = _shared_path(
            payload.get("output"),
            self.shared_root,
            suffix=".npy",
        )
        work_dir = _shared_path(payload.get("work_dir"), self.shared_root)
        seed_value = payload.get("seed")
        seed = None if seed_value is None else int(seed_value)
        loaded_features = np.load(features_path, allow_pickle=True)
        features = np.asarray(
            [
                np.asarray(feature_slice, dtype=np.float32)
                for feature_slice in loaded_features
            ],
            dtype=np.float32,
        )
        if features.ndim < 2 or len(features) < 1:
            raise ValueError("EDGE generation requires at least one feature slice")

        torch, model = self._runtime()
        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            np.random.seed(seed % (2**32 - 1))

        render_dir = work_dir / "edge_renders"
        motion_dir = work_dir / "edge_motions"
        shutil.rmtree(render_dir, ignore_errors=True)
        shutil.rmtree(motion_dir, ignore_errors=True)
        render_dir.mkdir(parents=True, exist_ok=True)
        motion_dir.mkdir(parents=True, exist_ok=True)
        condition = torch.from_numpy(features)
        model.render_sample(
            (None, condition, []),
            "test",
            str(render_dir),
            render_count=-1,
            fk_out=str(motion_dir),
            render=False,
        )
        outputs = sorted(glob.glob(str(motion_dir / "test_*.pkl")))
        if not outputs:
            raise RuntimeError(f"EDGE produced no motion in {motion_dir}")
        motion = self._pkl_to_edge151(Path(outputs[0]))
        expected_frames = int(5.0 * 30 + (len(features) - 1) * 2.5 * 30)
        if motion.shape[0] > expected_frames:
            motion = motion[:expected_frames]
        _atomic_save_array(output_path, motion)
        return {
            "output": str(output_path),
            "frames": int(motion.shape[0]),
            "summary": (
                f"EDGE long-form pipeline with {len(features)} chained 5s clips "
                f"and 2.5s overlap; output length {motion.shape[0]} frames."
            ),
            "seed": seed,
        }


class RenderFramesHandler:
    """Render one exact global frame range through a resident Blender daemon."""

    def __init__(
        self,
        *,
        shared_root: Path,
        width: int = 1080,
        height: int = 1080,
        samples: int = 96,
        engine: str = "eevee",
        denoise: int = 1,
        frame_format: str = "tga",
        daemon: int = 0,
        local_tmp: Path | None = None,
        worker_id: str = "",
    ):
        self.shared_root = Path(shared_root).resolve()
        self.width = max(1, int(width))
        self.height = max(1, int(height))
        self.samples = max(1, int(samples))
        self.engine = str(engine).lower()
        self.denoise = int(denoise)
        self.frame_format = str(frame_format).lower().lstrip(".")
        self.daemon = max(0, int(daemon))
        self.local_tmp = Path(
            local_tmp
            or os.environ.get("AGENTLODGE_WORKER_TMP", tempfile.gettempdir())
        ).resolve()
        raw_worker_id = (
            str(worker_id).strip()
            or os.environ.get("AGENTLODGE_WORKER_ID", "").strip()
            or self.local_tmp.name
            or "worker"
        )
        safe_worker_id = "".join(
            character
            if character.isalnum() or character in "_.-"
            else "_"
            for character in raw_worker_id
        )[:96]
        fallback_root = Path(
            os.environ.get(
                "AGENTLODGE_RENDER_FALLBACK_ROOT",
                tempfile.gettempdir(),
            )
        ).resolve()
        self.fallback_tmp = (
            fallback_root / f"agentlodge-render-{safe_worker_id}"
        ).resolve()
        if self.frame_format not in {"png", "tga"}:
            raise ValueError(f"unsupported render frame format: {self.frame_format}")

    def preload(self) -> None:
        from server import warm_render

        if not warm_render.on_pod():
            raise RuntimeError("render worker is missing local Blender assets")
        _ffmpeg_executable()
        ready = warm_render.ensure_pool(
            width=self.width,
            height=self.height,
            samples=self.samples,
            engine=self.engine,
            denoise=self.denoise,
            frame_format=self.frame_format,
            wait_ready=120,
        )
        if ready < 1:
            raise RuntimeError("render worker could not start a Blender daemon")

    def render_provenance(self) -> dict[str, Any]:
        from server import warm_render

        return warm_render.render_provenance()

    def _quality(self, fps: int) -> dict[str, Any]:
        return {
            "width": self.width,
            "height": self.height,
            "samples": self.samples,
            "engine": self.engine,
            "denoise": self.denoise,
            "frame_format": self.frame_format,
            "fps": int(fps),
        }

    def _scratch_bytes_required(self, frame_count: int) -> int:
        pixels = self.width * self.height
        source_bytes = frame_count * (pixels * 4 + 4096)
        ffv1_bytes = frame_count * pixels * 4
        working_bytes = source_bytes + ffv1_bytes
        return working_bytes + max(256 * 1024 * 1024, working_bytes // 2)

    def _shard_bytes_required(self, frame_count: int) -> int:
        return (
            frame_count * self.width * self.height * 4
            + 64 * 1024 * 1024
        )

    @staticmethod
    def _reservation_bytes() -> int | None:
        reservation = os.environ.get(
            "AGENTLODGE_SHM_RESERVATION_FILE",
            "",
        ).strip()
        if not reservation:
            return None
        try:
            fields = Path(reservation).read_text(encoding="utf-8").split()
            return int(fields[1])
        except (OSError, ValueError, IndexError):
            return None

    def _scratch_parent(self, required_bytes: int) -> tuple[Path, bool]:
        configured = self.local_tmp
        configured.mkdir(parents=True, exist_ok=True)
        shm_root = Path(
            os.environ.get("AGENTLODGE_SHM_ROOT", "/dev/shm")
        ).resolve()
        uses_shm = configured.is_relative_to(shm_root)
        free_bytes = shutil.disk_usage(configured).free
        reservation_bytes = self._reservation_bytes() if uses_shm else None
        if (
            uses_shm
            and (
                free_bytes < required_bytes
                or reservation_bytes is None
                or (
                    reservation_bytes < required_bytes
                )
            )
        ):
            fallback = self.fallback_tmp
            fallback.mkdir(parents=True, exist_ok=True)
            fallback_free = shutil.disk_usage(fallback).free
            if fallback_free < required_bytes:
                raise RuntimeError(
                    "render scratch estimate exceeds both shared memory and "
                    "worker fallback capacity"
                )
            logger.warning(
                "render task requires %d scratch bytes; falling back from %s "
                "to %s",
                required_bytes,
                configured,
                fallback,
            )
            return fallback, True
        if free_bytes < required_bytes:
            raise RuntimeError(
                f"render scratch requires {required_bytes} bytes but only "
                f"{free_bytes} bytes are free under {configured}"
            )
        return configured, False

    def __call__(self, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        from server import warm_render

        poses = _shared_path(
            payload.get("poses"),
            self.shared_root,
            must_exist=True,
            suffix=".npz",
        )
        shard_value = payload.get("shard_output")
        shard_output = (
            _shared_path(
                shard_value,
                self.shared_root,
                suffix=".mkv",
            )
            if shard_value
            else None
        )
        local_root = None
        frames_dir = None
        frame_start = max(0, int(payload.get("frame_start") or 0))
        frame_end = int(payload.get("frame_end"))
        if frame_end <= frame_start:
            raise ValueError("render frame_end must be greater than frame_start")
        expected = {
            "width": self.width,
            "height": self.height,
            "samples": self.samples,
            "engine": self.engine,
            "denoise": self.denoise,
            "frame_format": self.frame_format,
        }
        requested = {
            key: (
                str(payload.get(key)).lower().lstrip(".")
                if key in {"engine", "frame_format"}
                else int(payload.get(key))
            )
            for key in expected
        }
        if requested != expected:
            raise ValueError(
                f"render task quality mismatch: requested={requested}, worker={expected}"
            )
        requested_provenance = payload.get("render_provenance")
        local_provenance = self.render_provenance()
        fps = max(1, int(payload.get("fps") or 30))
        local_identity = render_identity_digest(
            local_provenance,
            self._quality(fps),
        )
        if (
            not isinstance(requested_provenance, Mapping)
            or dict(requested_provenance) != local_provenance
            or payload.get("render_contract_version")
            != RENDER_CONTRACT_VERSION
            or payload.get("render_identity_digest") != local_identity
        ):
            raise ValueError("render task provenance does not match this worker")
        if shard_output is not None:
            required_scratch_bytes = self._scratch_bytes_required(
                frame_end - frame_start
            )
            scratch_parent, scratch_fallback = self._scratch_parent(
                required_scratch_bytes
            )
            shard_output.parent.mkdir(parents=True, exist_ok=True)
            required_shard_bytes = self._shard_bytes_required(
                frame_end - frame_start
            )
            shard_free_bytes = shutil.disk_usage(shard_output.parent).free
            if shard_free_bytes < required_shard_bytes:
                raise RuntimeError(
                    f"render shard staging requires {required_shard_bytes} "
                    f"bytes but only {shard_free_bytes} bytes are free"
                )
            local_root = Path(
                tempfile.mkdtemp(
                    prefix="maestro-render-worker-",
                    dir=scratch_parent,
                )
            )
            frames_dir = local_root / "frames"
        else:
            required_scratch_bytes = 0
            scratch_fallback = False
            frames_dir = _shared_path(payload.get("frames_dir"), self.shared_root)
        assert frames_dir is not None
        ready = warm_render.ensure_pool(
            width=self.width,
            height=self.height,
            samples=self.samples,
            engine=self.engine,
            denoise=self.denoise,
            frame_format=self.frame_format,
            wait_ready=120,
        )
        if ready < 1:
            raise RuntimeError(
                "render worker could not restart an attested Blender daemon"
            )
        frames_dir.mkdir(parents=True, exist_ok=True)
        timeout = max(60.0, float(payload.get("timeout") or 900.0))
        try:
            ok = warm_render.warm_render(
                str(poses),
                str(frames_dir),
                daemon=self.daemon,
                samples=self.samples,
                width=self.width,
                height=self.height,
                engine=self.engine,
                denoise=self.denoise,
                fast=False,
                stride=1,
                batch_render=True,
                frame_start=frame_start,
                frame_end=frame_end,
                clear_frames=False,
                frame_format=self.frame_format,
                timeout=timeout,
            )
            if not ok:
                raise RuntimeError(
                    f"Blender daemon failed frame range [{frame_start}, {frame_end})"
                )
            daemon_attestation = warm_render.daemon_attestation(
                self.daemon,
                width=self.width,
                height=self.height,
                samples=self.samples,
                engine=self.engine,
                denoise=self.denoise,
                frame_format=self.frame_format,
            )
            attested_provenance = {
                key: daemon_attestation.get(key)
                for key in (
                    "render_contract_version",
                    "daemon_protocol_version",
                    "scene",
                    "renderer",
                )
            }
            selector = daemon_attestation.get("selector")
            attested_provenance["selector"] = (
                None
                if selector is None
                else {
                    key: selector.get(key)
                    for key in ("version", "build_id", "binary_sha256")
                }
            )
            if attested_provenance != local_provenance:
                raise RuntimeError(
                    "Blender daemon attestation does not match worker provenance"
                )
            missing = [
                frame
                for frame in range(frame_start, frame_end)
                if not (
                    frames_dir / f"frame_{frame:04d}.{self.frame_format}"
                ).is_file()
            ]
            if missing:
                raise RuntimeError(
                    f"render worker omitted {len(missing)} frames; "
                    f"first missing={missing[0]}"
                )
            output = {
                "frame_start": frame_start,
                "frame_end": frame_end,
                "frames": frame_end - frame_start,
                "source_frames_sha256": _frame_digest(
                    frames_dir,
                    frame_start,
                    frame_end,
                    self.frame_format,
                ),
                "render_contract_version": RENDER_CONTRACT_VERSION,
                "render_provenance": local_provenance,
                "render_identity_digest": local_identity,
                "daemon_attestation": daemon_attestation,
                "scratch": {
                    "estimated_bytes": required_scratch_bytes,
                    "fallback_from_shm": scratch_fallback,
                },
                **expected,
            }
            if shard_output is not None:
                source_rgb_sha256 = source_sequence_rgb_sha256(
                    frames_dir,
                    frame_start=frame_start,
                    frame_end=frame_end,
                    frame_format=self.frame_format,
                    width=self.width,
                    height=self.height,
                    fps=fps,
                )
                local_shard = local_root / "shard.mkv"
                _package_ffv1(
                    frames_dir,
                    local_shard,
                    frame_start=frame_start,
                    frame_end=frame_end,
                    frame_format=self.frame_format,
                    fps=fps,
                )
                shard_validation = probe_ffv1_shard(
                    local_shard,
                    frame_start=frame_start,
                    frame_end=frame_end,
                    width=self.width,
                    height=self.height,
                    fps=fps,
                )
                shard_validation.update(
                    {
                        "decoded_rgb_digest_version": RGB_DIGEST_VERSION,
                        "decoded_rgb_sha256": source_rgb_sha256,
                        "worker_validation_version": (
                            WORKER_SHARD_VALIDATION_VERSION
                        ),
                        "worker_shard_full_decode": False,
                    }
                )
                shard_output.parent.mkdir(parents=True, exist_ok=True)
                shared_temporary = shard_output.with_name(
                    shard_output.name + f".{uuid.uuid4().hex}.tmp"
                )
                try:
                    shutil.copy2(local_shard, shared_temporary)
                    os.replace(shared_temporary, shard_output)
                finally:
                    shared_temporary.unlink(missing_ok=True)
                shard_digest = hashlib.sha256()
                with shard_output.open("rb") as handle:
                    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                        shard_digest.update(chunk)
                output.update(
                    {
                        "shard_output": str(shard_output),
                        "shard_sha256": shard_digest.hexdigest(),
                        "transport": "ffv1",
                        "fps": fps,
                        "source_decoded_rgb_sha256": source_rgb_sha256,
                        "shard_decoded_rgb_sha256": source_rgb_sha256,
                        "decoded_rgb_digest_version": RGB_DIGEST_VERSION,
                        "shard_validation": shard_validation,
                    }
                )
            return output
        finally:
            if local_root is not None:
                shutil.rmtree(local_root, ignore_errors=True)
