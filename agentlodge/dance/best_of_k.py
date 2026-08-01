"""Best-of-K seed selection for the diffusion dance generators (EDGE / LODGE).

Both generators are seed-stochastic and neither explicitly optimizes beat alignment, so beat sync
varies noticeably across seeds. This module samples K candidates for a musical section and selects
the one that best matches a composite objective dominated by **Beat Alignment Score** (the
pipeline's main weakness), following the inference-time-scaling result that random best-of-N with a
task verifier is a strong, low-risk use of extra compute (Ma et al., 2025, arXiv:2501.09732).

It is **generator-agnostic**: pass a ``generate_fn(seed) -> (L, 139) motion`` closure that wraps a
real seeded LODGE or EDGE call, so the scoring/selection logic here is fully unit-testable without
the heavy models. Candidates are independent -> the K generations can be run in parallel.
"""

from __future__ import annotations

import logging
from typing import Callable, Sequence

import numpy as np

from agentlodge.dance.beat_metrics import (
    _kinematic_speed,
    beat_alignment_score,
    foot_contact_consistency,
)

logger = logging.getLogger(__name__)

# Composite weights: BAS dominates (the weakness); foot-contact + energy match are secondary.
_W_BAS = 0.6
_W_FOOT = 0.25
_W_ENERGY = 0.15


def _mean_speed(motion: np.ndarray) -> float:
    return float(np.mean(_kinematic_speed(motion)))


# Instruction-aware reward weighting for the interactive editor (agentlodge.editor). Maps a parsed
# edit objective to best-of-K composite weights + an energy target, so windowed regeneration selects
# the candidate that best realizes the user's intent ("more energetic" weights the energy term and
# targets high energy; "more on beat" weights BAS; "calmer" targets low energy).
_OBJECTIVE_WEIGHTS = {
    "more_on_beat":   {"w_bas": 0.80, "w_foot": 0.10, "w_energy": 0.10, "target_intensity": None},
    "more_energetic": {"w_bas": 0.20, "w_foot": 0.10, "w_energy": 0.70, "target_intensity": 1.0},
    "sharper":        {"w_bas": 0.30, "w_foot": 0.10, "w_energy": 0.60, "target_intensity": 1.0},
    "calmer":         {"w_bas": 0.20, "w_foot": 0.15, "w_energy": 0.65, "target_intensity": 0.0},
    "smoother":       {"w_bas": 0.20, "w_foot": 0.35, "w_energy": 0.45, "target_intensity": 0.0},
}


def objective_weights(objective: str) -> dict:
    """Best-of-K composite weights + energy target for an editor objective (falls back to defaults)."""
    return dict(_OBJECTIVE_WEIGHTS.get(
        objective,
        {"w_bas": _W_BAS, "w_foot": _W_FOOT, "w_energy": _W_ENERGY, "target_intensity": None},
    ))


def score_candidates(motions: Sequence[np.ndarray], music_beat_frames, *,
                     target_intensity: float | None = None, sigma_frames: float = 3.0,
                     w_bas: float = _W_BAS, w_foot: float = _W_FOOT,
                     w_energy: float = _W_ENERGY, score_transform=None) -> list[dict]:
    """Score each candidate motion. Energy match is computed *across the candidate set* (min-max),
    so ``target_intensity`` in [0,1] means "prefer the livelier / calmer candidate".

    ``score_transform`` optionally maps each raw candidate to the AgentLODGE 139-dim scoring format
    (e.g. a raw LODGE/EDGE output -> ``to_zup(to_agentlodge139(ensure_lodge139(m)))``) before scoring.
    """
    prepped = [score_transform(m) if score_transform is not None else m for m in motions]
    n = len(prepped)
    bas = [beat_alignment_score(m, music_beat_frames, sigma_frames=sigma_frames) for m in prepped]
    foot = [foot_contact_consistency(m) for m in prepped]
    speeds = np.array([_mean_speed(m) for m in prepped], dtype=np.float64)
    if target_intensity is not None and n > 1 and speeds.max() > speeds.min():
        erel = (speeds - speeds.min()) / (speeds.max() - speeds.min())
        ems = 1.0 - np.abs(erel - float(target_intensity))
    else:
        ems = np.ones(n)
    out = []
    for i in range(n):
        total = w_bas * bas[i] + w_foot * foot[i] + w_energy * float(ems[i])
        out.append({
            "bas": round(float(bas[i]), 4),
            "foot_contact_consistency": round(float(foot[i]), 4),
            "energy_match": round(float(ems[i]), 4),
            "total": round(float(total), 4),
        })
    return out


def select_best(motions: Sequence[np.ndarray], music_beat_frames, **kw) -> tuple[int, list[dict]]:
    """Return ``(best_index, per_candidate_scores)``, the argmax of the composite score."""
    if not motions:
        raise ValueError("select_best: no candidates")
    scores = score_candidates(motions, music_beat_frames, **kw)
    best = int(np.argmax([s["total"] for s in scores]))
    return best, scores


def best_of_k(generate_fn: Callable[[int], np.ndarray | None], seeds: Sequence[int],
              music_beat_frames, *, target_intensity: float | None = None,
              min_frames: int = 3, **score_kw) -> tuple[np.ndarray, int, dict]:
    """Generate one candidate per seed via ``generate_fn(seed)``, score, and return the best.

    Returns ``(best_motion, best_seed, report)``. ``report`` records every seed's scores and the
    winner. Invalid / too-short / failed generations (``None``) are skipped. Raises if none succeed.
    """
    motions: list[np.ndarray] = []
    used: list[int] = []
    for s in seeds:
        try:
            m = generate_fn(int(s))
        except Exception:  # noqa: BLE001 - a bad seed shouldn't kill the whole search
            m = None
        if m is not None and np.asarray(m).ndim == 2 and np.asarray(m).shape[0] >= min_frames:
            motions.append(np.asarray(m, dtype=np.float32))
            used.append(int(s))
    if not motions:
        raise ValueError("best_of_k: no valid candidates were generated")
    best, scores = select_best(motions, music_beat_frames, target_intensity=target_intensity,
                               **score_kw)
    report = {
        "k": len(motions),
        "seeds": used,
        "scores": scores,
        "winner_index": best,
        "winner_seed": used[best],
        "winner_bas": scores[best]["bas"],
    }
    return motions[best], used[best], report


def generate_best_of_k(generate_fn: Callable[[int], np.ndarray | None], k: int,
                       music_beat_frames, *, base_seed: int = 0, **score_kw):
    """Convenience: run best-of-K over consecutive seeds ``base_seed .. base_seed+k-1``.

    ``score_kw`` may include ``target_intensity`` and ``score_transform`` (to convert a raw
    generator output to the 139-dim scoring format before BAS). Returns ``(motion, seed, report)``.
    """
    seeds = list(range(int(base_seed), int(base_seed) + max(1, int(k))))
    return best_of_k(generate_fn, seeds, music_beat_frames, **score_kw)


def best_of_k_job(job_fn: Callable[[int | None], dict], k: int | None, music_beat_frames, *,
                  score_transform=None) -> dict:
    """Run a generation ``job_fn(seed) -> {"motion", "error", "summary", ...}`` under best-of-K
    beat-alignment selection.

    Generates K seeded candidates (seeds 0..K-1), scores each by BAS (after ``score_transform`` to
    the 139-dim scoring format), and returns the winning job dict (summary annotated). Falls back to
    a single ungated run (``job_fn(None)``) when K<=1, no beats are available, or selection fails.
    """
    if not k or int(k) <= 1 or music_beat_frames is None:
        return job_fn(None)
    results: dict[int, dict] = {}

    def gen(seed):
        r = job_fn(seed)
        results[seed] = r
        return r.get("motion") if r.get("error") is None else None

    try:
        _, seed, report = best_of_k(gen, list(range(int(k))), music_beat_frames,
                                    score_transform=score_transform)
        r = dict(results[seed])
        r["summary"] = (r.get("summary", "")
                        + f" | best-of-{report['k']} by BAS: seed {seed} (BAS {report['winner_bas']})")
        logger.info("Best-of-%s selected seed %s (BAS %s)", report["k"], seed, report["winner_bas"])
        return r
    except Exception as exc:  # noqa: BLE001 - never let selection break generation
        logger.warning("best-of-K selection failed (%s); using a single generation", exc)
        for r in results.values():
            if r.get("error") is None and r.get("motion") is not None:
                return r
        return job_fn(None)
