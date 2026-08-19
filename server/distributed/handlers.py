"""GPU task handlers used by persistent RunPod workers."""

from __future__ import annotations

import glob
import hashlib
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
    ):
        self.edge_root = Path(edge_root).resolve()
        self.shared_root = Path(shared_root).resolve()
        self._extractor = extractor
        configured_seconds = (
            os.environ.get("AGENTLODGE_JUKEBOX_WARMUP_SECONDS", "5")
            if preload_audio_seconds is None
            else preload_audio_seconds
        )
        self.preload_audio_seconds = max(0.0, float(configured_seconds))

    def _load_extractor(self) -> Callable[[str], tuple[Any, Any]]:
        if self._extractor is not None:
            return self._extractor
        if not self.edge_root.is_dir():
            raise FileNotFoundError(f"EDGE root does not exist: {self.edge_root}")
        os.chdir(self.edge_root)
        if str(self.edge_root) not in sys.path:
            sys.path.insert(0, str(self.edge_root))
        from data.audio_extraction.jukebox_features import extract

        self._extractor = extract
        return extract

    def preload(self) -> None:
        extractor = self._load_extractor()
        import jukemirlib

        if jukemirlib.VQVAE is None and jukemirlib.TOP_PRIOR is None:
            jukemirlib.VQVAE, jukemirlib.TOP_PRIOR = jukemirlib.setup_models()
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
                representations, _ = extractor(str(audio_path))
            finally:
                os.chdir(previous_directory)
        warmed = np.asarray(representations)
        if warmed.size == 0 or not np.isfinite(warmed).all():
            raise RuntimeError("Jukebox preload inference produced invalid features")

    def __call__(self, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        items = payload.get("items")
        if not isinstance(items, list) or not items:
            raise ValueError("jukebox.extract requires a non-empty items list")
        extractor = self._load_extractor()
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
            representations, _ = extractor(str(wav_path))
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
            wait_ready=120,
        )
        if ready < 1:
            raise RuntimeError("render worker could not start a Blender daemon")

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
        if shard_output is not None:
            self.local_tmp.mkdir(parents=True, exist_ok=True)
            local_root = Path(
                tempfile.mkdtemp(
                    prefix="maestro-render-worker-",
                    dir=self.local_tmp,
                )
            )
            frames_dir = local_root / "frames"
        else:
            frames_dir = _shared_path(payload.get("frames_dir"), self.shared_root)
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
                **expected,
            }
            if shard_output is not None:
                fps = max(1, int(payload.get("fps") or 30))
                local_shard = local_root / "shard.mkv"
                _package_ffv1(
                    frames_dir,
                    local_shard,
                    frame_start=frame_start,
                    frame_end=frame_end,
                    frame_format=self.frame_format,
                    fps=fps,
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
                    }
                )
            return output
        finally:
            if local_root is not None:
                shutil.rmtree(local_root, ignore_errors=True)
