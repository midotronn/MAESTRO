#!/usr/bin/env python3
"""Generate the canonical full-song LODGE, EDGE, and story motions from cached audio features."""

from __future__ import annotations

import hashlib
import json
import os
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import librosa
import numpy as np

WORKSPACE = Path(os.environ.get("WORKSPACE", "/workspace")).resolve()
AGENTLODGE_ROOT = Path(
    os.environ.get("AGENTLODGE_ROOT", str(WORKSPACE / "AgentLODGE"))
).resolve()
sys.path.insert(0, str(AGENTLODGE_ROOT))

from agentlodge.agent.storyboard import author_storyboard  # noqa: E402
from agentlodge.audio.preprocess import (  # noqa: E402
    extract_audio_descriptor,
    extract_song_metadata,
    release_torch_memory,
)
from agentlodge.audio.structure import analyze_structure  # noqa: E402
from agentlodge.config import Settings  # noqa: E402
from agentlodge.dance.best_of_k import best_of_k_job  # noqa: E402
from agentlodge.dance.story import build_story_dance  # noqa: E402
from agentlodge.pipeline import (  # noqa: E402
    _edge_score_transform,
    _lodge_score_transform,
    _run_edge_job,
    _run_lodge_job,
    _settings_to_dict,
    _use_parallel_execution,
)
from server.distributed.runtime import capability_enabled  # noqa: E402

_TIMING_LOCK = threading.Lock()


def _emit_timing(stage: str, state: str, detail: str = "") -> None:
    timestamp_ms = time.time_ns() // 1_000_000
    line = f"MAESTRO_TIMING {stage} {state} {timestamp_ms} {detail}".rstrip()
    print(line, flush=True)
    timing_file = os.environ.get("MAESTRO_TIMING_FILE", "").strip()
    if not timing_file:
        return
    with _TIMING_LOCK:
        path = Path(timing_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")


def _settings() -> Settings:
    return Settings.from_dict(
        {
            "lodge_code_path": str(WORKSPACE / "LODGE"),
            "lodge_weights_path": str(
                WORKSPACE
                / "LODGE/exp/Local_Module/FineDance_FineTuneV2_Local/checkpoints/epoch=299.ckpt"
            ),
            "lodge_global_weights_path": str(
                WORKSPACE
                / "LODGE/exp/Global_Module/FineDance_Global/checkpoints/epoch=2999.ckpt"
            ),
            "edge_code_path": str(WORKSPACE / "EDGE"),
            "edge_weights_path": str(WORKSPACE / "EDGE/checkpoint.pt"),
            "lodge_genre": "Hiphop",
            "max_edge_slices": None,
        }
    )


def _successful_motion(result: dict, label: str) -> np.ndarray:
    motion = result.get("motion")
    if result.get("error") or motion is None:
        detail = str(result.get("error") or "no motion returned").splitlines()[0]
        raise RuntimeError(f"{label} generation failed: {detail}")
    return np.asarray(motion, dtype=np.float32)


def _file_fingerprint(path: Path) -> dict[str, object]:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return {
        "bytes": path.stat().st_size,
        "sha256": digest.hexdigest(),
    }


def _distributed_generation_job(
    backbone: str,
    features_path: Path,
    output_path: Path,
    work_dir: Path,
    *,
    seed: int | None,
) -> dict:
    capability = f"{backbone}.generate"
    try:
        from server.distributed import FileTaskCoordinator, WorkerRegistry

        registry = WorkerRegistry.from_env()
        coordinator = FileTaskCoordinator(
            registry,
            heartbeat_max_age=float(
                os.environ.get("AGENTLODGE_WORKER_HEARTBEAT_MAX_AGE", "30")
            ),
        )
        handle = coordinator.submit(
            capability,
            {
                "features": str(features_path.resolve()),
                "output": str(output_path.resolve()),
                "work_dir": str(work_dir.resolve()),
                "seed": seed,
                "source": _file_fingerprint(features_path),
            },
        )
        result = coordinator.wait(
            handle,
            timeout=float(
                os.environ.get("AGENTLODGE_GENERATION_TIMEOUT", "1200")
            ),
        )
        return {
            "motion": np.load(output_path).astype(np.float32),
            "summary": str(result.output.get("summary") or ""),
            "error": None,
            "worker_id": result.worker_id,
        }
    except Exception as exc:  # noqa: BLE001 - match the existing job result contract
        return {
            "motion": None,
            "summary": "",
            "error": f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}",
        }


def generate_song(sid: str) -> dict:
    wav = WORKSPACE / f"LODGE/data/finedance/music_wav/{sid}.wav"
    lodge_features_path = WORKSPACE / f"lodge_fd_{sid}_feats.npy"
    edge_slices_path = WORKSPACE / f"edge{sid}_slices.npy"
    for path in (wav, lodge_features_path, edge_slices_path):
        if not path.is_file():
            raise FileNotFoundError(f"missing required input: {path}")

    settings = _settings()
    settings_dict = _settings_to_dict(settings)
    metadata = extract_song_metadata(wav)
    if metadata.duration_seconds < settings.min_audio_seconds:
        raise ValueError(
            f"audio is {metadata.duration_seconds:.1f}s; "
            f"at least {settings.min_audio_seconds:.0f}s is required"
        )
    k = max(1, int(os.environ.get("AGENTLODGE_BEST_OF_K", "1")))
    work = WORKSPACE / f"gen{sid}_work"
    lodge_features = np.load(lodge_features_path).astype(np.float32)
    edge_slices = [
        np.asarray(item, dtype=np.float32)
        for item in np.load(edge_slices_path, allow_pickle=True)
    ]

    def generate_lodge() -> dict:
        _emit_timing("generation_lodge", "start", f"k={k}")
        result = None
        try:
            result = best_of_k_job(
                lambda seed: (
                    _distributed_generation_job(
                        "lodge",
                        lodge_features_path,
                        work
                        / "lodge"
                        / (f"seed_{seed}" if seed is not None else "single")
                        / "lodge_motion.npy",
                        work
                        / "lodge"
                        / (f"seed_{seed}" if seed is not None else "single"),
                        seed=seed,
                    )
                    if capability_enabled("lodge.generate")
                    else _run_lodge_job(
                        lodge_features,
                        settings_dict,
                        str(
                            work
                            / "lodge"
                            / (f"seed_{seed}" if seed is not None else "single")
                        ),
                        seed=seed,
                    )
                ),
                k,
                metadata.beat_frames,
                score_transform=_lodge_score_transform,
            )
            return result
        finally:
            worker = (
                f" worker={result.get('worker_id')}"
                if result and result.get("worker_id")
                else ""
            )
            _emit_timing("generation_lodge", "end", f"k={k}{worker}")

    def generate_edge() -> dict:
        _emit_timing("generation_edge", "start", f"k={k}")
        result = None
        try:
            result = best_of_k_job(
                lambda seed: (
                    _distributed_generation_job(
                        "edge",
                        edge_slices_path,
                        work
                        / "edge"
                        / (f"seed_{seed}" if seed is not None else "single")
                        / "edge_motion.npy",
                        work
                        / "edge"
                        / (f"seed_{seed}" if seed is not None else "single"),
                        seed=seed,
                    )
                    if capability_enabled("edge.generate")
                    else _run_edge_job(
                        str(wav),
                        edge_slices,
                        settings_dict,
                        str(
                            work
                            / "edge"
                            / (f"seed_{seed}" if seed is not None else "single")
                        ),
                        seed=seed,
                    )
                ),
                k,
                metadata.beat_frames,
                score_transform=_edge_score_transform,
            )
            return result
        finally:
            worker = (
                f" worker={result.get('worker_id')}"
                if result and result.get("worker_id")
                else ""
            )
            _emit_timing("generation_edge", "end", f"k={k}{worker}")

    if _use_parallel_execution():
        print(f"[{sid}] generating LODGE and EDGE in parallel", flush=True)
        print(
            "MAESTRO_SUBPROGRESS generation 5 Starting both motion backbones",
            flush=True,
        )
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = {
                executor.submit(generate_lodge): "LODGE",
                executor.submit(generate_edge): "EDGE",
            }
            generated = {}
            for completed, future in enumerate(as_completed(futures), start=1):
                label = futures[future]
                generated[label] = future.result()
                progress = 45 if completed == 1 else 80
                print(
                    f"MAESTRO_SUBPROGRESS generation {progress} "
                    f"{label} motion generation complete",
                    flush=True,
                )
            lodge_result = generated["LODGE"]
            edge_result = generated["EDGE"]
        if lodge_result.get("error") or lodge_result.get("motion") is None:
            print(f"[{sid}] parallel LODGE failed; retrying it alone", flush=True)
            print(
                "MAESTRO_SUBPROGRESS generation 45 Retrying LODGE generation",
                flush=True,
            )
            lodge_result = generate_lodge()
        if edge_result.get("error") or edge_result.get("motion") is None:
            print(f"[{sid}] parallel EDGE failed; retrying it alone", flush=True)
            print(
                "MAESTRO_SUBPROGRESS generation 45 Retrying EDGE generation",
                flush=True,
            )
            edge_result = generate_edge()
    else:
        print(
            "MAESTRO_SUBPROGRESS generation 5 Generating LODGE motion",
            flush=True,
        )
        lodge_result = generate_lodge()
        print(
            "MAESTRO_SUBPROGRESS generation 45 LODGE motion generation complete",
            flush=True,
        )
        edge_result = generate_edge()
        print(
            "MAESTRO_SUBPROGRESS generation 80 EDGE motion generation complete",
            flush=True,
        )

    lodge_motion = _successful_motion(lodge_result, "LODGE")
    edge_motion = _successful_motion(edge_result, "EDGE")
    np.save(WORKSPACE / f"lodge_fd_{sid}_full.npy", lodge_motion)
    np.save(WORKSPACE / f"edge_fd_{sid}_full.npy", edge_motion)
    release_torch_memory()

    print(
        "MAESTRO_SUBPROGRESS generation 84 Analyzing the song structure",
        flush=True,
    )
    total_frames = min(lodge_motion.shape[0], edge_motion.shape[0])
    _emit_timing("structure_analysis", "start")
    try:
        structure = analyze_structure(
            wav,
            metadata,
            total_frames,
            min_section_seconds=8.0,
        )
    finally:
        _emit_timing("structure_analysis", "end")
    _emit_timing("audio_descriptor", "start")
    try:
        waveform, sample_rate = librosa.load(str(wav), sr=22050, mono=True)
        descriptor = extract_audio_descriptor(waveform, sample_rate, metadata)
    finally:
        _emit_timing("audio_descriptor", "end")
    print(
        "MAESTRO_SUBPROGRESS generation 90 Authoring the choreography storyboard",
        flush=True,
    )
    _emit_timing("storyboard", "start")
    try:
        storyboard = author_storyboard(
            structure,
            metadata,
            descriptor,
            api_key=os.environ.get("OPENAI_API_KEY") or None,
            motif_reuse=True,
        )
    finally:
        _emit_timing("storyboard", "end")
    print(
        "MAESTRO_SUBPROGRESS generation 95 Assembling the final choreography",
        flush=True,
    )
    _emit_timing("story_assembly", "start")
    try:
        assembled = build_story_dance(
            lodge_motion,
            edge_motion,
            structure,
            storyboard,
            metadata,
            blend_frames=15,
            motif_reuse=True,
        )
    finally:
        _emit_timing("story_assembly", "end")
    motion = assembled.motion
    schedule = [
        [int(a), int(b), str(source), str(role)]
        for a, b, source, role in assembled.schedule
    ]

    output = WORKSPACE / f"fd_{sid}_STORY_bestofk.npy"
    np.save(output, np.asarray(motion, dtype=np.float32))
    print(
        "MAESTRO_SUBPROGRESS generation 100 Motion generation complete",
        flush=True,
    )

    report = {
        "sid": sid,
        "best_of_k": k,
        "frames": int(motion.shape[0]),
        "lodge_summary": lodge_result.get("summary", ""),
        "edge_summary": edge_result.get("summary", ""),
        "generation_workers": {
            "lodge": lodge_result.get("worker_id"),
            "edge": edge_result.get("worker_id"),
        },
        "reasoning": assembled.reasoning,
        "schedule": schedule,
        "storyboard": storyboard.to_dict(),
        "section_scores": assembled.section_scores,
    }
    (WORKSPACE / f"fd_{sid}_STORY_bestofk.json").write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: make_song_bestofk.py <sid>", file=sys.stderr)
        return 2
    report = generate_song(sys.argv[1])
    print(
        f"MAKE_SONG_{report['sid']}_DONE "
        f"{report['frames']} frames best-of-{report['best_of_k']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
