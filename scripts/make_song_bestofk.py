#!/usr/bin/env python3
"""Generate the canonical full-song LODGE, EDGE, and story motions from cached audio features."""

from __future__ import annotations

import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
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
        return best_of_k_job(
            lambda seed: _run_lodge_job(
                lodge_features,
                settings_dict,
                str(work / "lodge" / (f"seed_{seed}" if seed is not None else "single")),
                seed=seed,
            ),
            k,
            metadata.beat_frames,
            score_transform=_lodge_score_transform,
        )

    def generate_edge() -> dict:
        return best_of_k_job(
            lambda seed: _run_edge_job(
                str(wav),
                edge_slices,
                settings_dict,
                str(work / "edge" / (f"seed_{seed}" if seed is not None else "single")),
                seed=seed,
            ),
            k,
            metadata.beat_frames,
            score_transform=_edge_score_transform,
        )

    if _use_parallel_execution():
        print(f"[{sid}] generating LODGE and EDGE in parallel", flush=True)
        with ThreadPoolExecutor(max_workers=2) as executor:
            lodge_future = executor.submit(generate_lodge)
            edge_future = executor.submit(generate_edge)
            lodge_result = lodge_future.result()
            edge_result = edge_future.result()
        if lodge_result.get("error") or lodge_result.get("motion") is None:
            print(f"[{sid}] parallel LODGE failed; retrying it alone", flush=True)
            lodge_result = generate_lodge()
        if edge_result.get("error") or edge_result.get("motion") is None:
            print(f"[{sid}] parallel EDGE failed; retrying it alone", flush=True)
            edge_result = generate_edge()
    else:
        lodge_result = generate_lodge()
        edge_result = generate_edge()

    lodge_motion = _successful_motion(lodge_result, "LODGE")
    edge_motion = _successful_motion(edge_result, "EDGE")
    np.save(WORKSPACE / f"lodge_fd_{sid}_full.npy", lodge_motion)
    np.save(WORKSPACE / f"edge_fd_{sid}_full.npy", edge_motion)
    release_torch_memory()

    total_frames = min(lodge_motion.shape[0], edge_motion.shape[0])
    structure = analyze_structure(
        wav,
        metadata,
        total_frames,
        min_section_seconds=8.0,
    )
    waveform, sample_rate = librosa.load(str(wav), sr=22050, mono=True)
    descriptor = extract_audio_descriptor(waveform, sample_rate, metadata)
    storyboard = author_storyboard(
        structure,
        metadata,
        descriptor,
        api_key=os.environ.get("OPENAI_API_KEY") or None,
        motif_reuse=True,
    )
    assembled = build_story_dance(
        lodge_motion,
        edge_motion,
        structure,
        storyboard,
        metadata,
        blend_frames=15,
        motif_reuse=True,
    )
    motion = assembled.motion
    schedule = [
        [int(a), int(b), str(source), str(role)]
        for a, b, source, role in assembled.schedule
    ]

    output = WORKSPACE / f"fd_{sid}_STORY_bestofk.npy"
    np.save(output, np.asarray(motion, dtype=np.float32))

    report = {
        "sid": sid,
        "best_of_k": k,
        "frames": int(motion.shape[0]),
        "lodge_summary": lodge_result.get("summary", ""),
        "edge_summary": edge_result.get("summary", ""),
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
