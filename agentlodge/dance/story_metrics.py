"""Structure / "story" quality metrics for an assembled dance.

No standard metric exists for dance *structure*, so these quantify the properties the storyboard
stage is designed to produce (all computed directly from the Z-up 139-dim motion + the detected
:class:`~agentlodge.audio.structure.MusicStructure`, no forward kinematics, no librosa):

  * ``arc_adherence``     -- correlation between the dance's per-frame kinematic energy and the
                             song's energy arc (higher = the dance builds/resolves with the music).
  * ``sectional_contrast``-- mean pose distance between different-label sections (higher = parts
                             feel distinct).
  * ``motif_recurrence``  -- pose similarity within same-label sections (higher = motifs recur).
  * ``boundary_alignment``-- fraction of motion novelty peaks near musical section boundaries.
  * ``peak_jerk`` / ``area_under_jerk`` -- FlowMDM transition-quality metrics around section seams
                             (lower = smoother joins).
"""

from __future__ import annotations

import numpy as np

from agentlodge.config import FPS

_KIN = 135
_EPS = 1e-8


def _frame_energy(motion: np.ndarray) -> np.ndarray:
    root_vel = np.linalg.norm(np.diff(motion[:, :3], axis=0, prepend=motion[:1, :3]), axis=1)
    joint = np.linalg.norm(np.diff(motion[:, :_KIN], axis=0, prepend=motion[:1, :_KIN]), axis=1)
    return 0.6 * root_vel + 0.4 * joint


def _norm01(x: np.ndarray) -> np.ndarray:
    lo, hi = float(np.min(x)), float(np.max(x))
    return (x - lo) / (hi - lo) if hi > lo else np.zeros_like(x)


def arc_adherence(motion: np.ndarray, energy_curve: np.ndarray) -> float:
    """Pearson correlation of (smoothed) dance energy with the song energy arc, in [-1, 1]."""
    L = motion.shape[0]
    if L < 4 or energy_curve.size < 2:
        return 0.0
    de = _frame_energy(motion)
    win = max(1, FPS // 2)
    de = np.convolve(de, np.ones(win) / win, mode="same")
    target = np.interp(np.linspace(0, 1, L), np.linspace(0, 1, energy_curve.size), energy_curve)
    a, b = _norm01(de), _norm01(target)
    if a.std() < _EPS or b.std() < _EPS:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def _section_features(motion: np.ndarray, sections: list) -> list[np.ndarray]:
    return [motion[s.start_frame:s.end_frame, :_KIN].mean(axis=0)
            for s in sections if s.end_frame > s.start_frame]


def _resampled_kinematics(segment: np.ndarray, frames: int = 64) -> np.ndarray:
    """Resample a section and remove its initial pose so static identity channels cannot dominate."""
    source = np.asarray(segment[:, :_KIN], dtype=np.float64)
    if source.shape[0] < 2:
        return np.zeros((frames, _KIN), dtype=np.float64)
    old_time = np.linspace(0.0, 1.0, source.shape[0])
    new_time = np.linspace(0.0, 1.0, frames)
    sampled = np.stack(
        [np.interp(new_time, old_time, source[:, channel]) for channel in range(_KIN)],
        axis=1,
    )
    return sampled - sampled[:1]


def _cosine_motion_similarity(first: np.ndarray, second: np.ndarray) -> float:
    a = _resampled_kinematics(first).reshape(-1)
    b = _resampled_kinematics(second).reshape(-1)
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denominator < _EPS:
        return 0.0
    return float(np.clip(np.dot(a, b) / denominator, -1.0, 1.0))


def _motif_similarity(first: np.ndarray, second: np.ndarray) -> float:
    """Best dynamic similarity under bounded phase shift, mirror, and retrograde transforms."""
    from agentlodge.dance.transition import mirror, retrograde

    candidates = [
        second,
        retrograde(second),
        mirror(second),
        retrograde(mirror(second)),
    ]
    reference = _resampled_kinematics(first)
    reference_flat = reference.reshape(-1)
    reference_norm = float(np.linalg.norm(reference_flat))
    if reference_norm < _EPS:
        return 0.0
    best = -1.0
    max_shift = max(1, reference.shape[0] // 8)
    for candidate in candidates:
        sampled = _resampled_kinematics(candidate)
        for shift in range(-max_shift, max_shift + 1):
            shifted = np.roll(sampled, shift, axis=0)
            shifted = shifted - shifted[:1]
            flat = shifted.reshape(-1)
            denominator = reference_norm * float(np.linalg.norm(flat))
            if denominator < _EPS:
                continue
            best = max(
                best,
                float(np.clip(np.dot(reference_flat, flat) / denominator, -1.0, 1.0)),
            )
    return max(0.0, best)


def sectional_contrast(motion: np.ndarray, sections: list) -> float:
    """Mean pose distance between sections with DIFFERENT repetition labels (higher = distinct)."""
    feats = _section_features(motion, sections)
    labels = [s.label for s in sections if s.end_frame > s.start_frame]
    dists = [float(np.linalg.norm(feats[i] - feats[j]))
             for i in range(len(feats)) for j in range(i + 1, len(feats))
             if labels[i] != labels[j]]
    return float(np.mean(dists)) if dists else 0.0


def motif_recurrence(motion: np.ndarray, sections: list) -> float:
    """Dynamic same-label similarity invariant to mirror, retrograde, and small phase shifts.

    Sections are resampled, expressed relative to their initial pose, and compared under the
    choreographic transformations MAESTRO intentionally uses for motif variation. A fully frozen
    section has no dynamic signature and scores 0 rather than receiving false credit for repetition.
    """
    segments = [
        motion[s.start_frame:s.end_frame]
        for s in sections
        if s.end_frame > s.start_frame
    ]
    labels = [s.label for s in sections if s.end_frame > s.start_frame]
    similarities = [
        _motif_similarity(segments[i], segments[j])
        for i in range(len(segments))
        for j in range(i + 1, len(segments))
        if labels[i] == labels[j]
    ]
    return float(np.mean(similarities)) if similarities else 0.0


def boundary_alignment(motion: np.ndarray, sections: list, tol_seconds: float = 0.5) -> float:
    """Fraction of motion novelty peaks within ``tol_seconds`` of a section boundary."""
    from scipy.signal import find_peaks

    L = motion.shape[0]
    if L < 8 or len(sections) < 2:
        return 0.0
    energy = _frame_energy(motion)
    novelty = np.abs(np.diff(energy, prepend=energy[:1]))
    if novelty.max() <= _EPS:
        return 0.0
    peaks, _ = find_peaks(novelty, height=float(np.percentile(novelty, 75)),
                          distance=max(1, FPS // 2))
    if peaks.size == 0:
        return 0.0
    bounds = np.array([s.start_frame for s in sections[1:]], dtype=np.int64)
    tol = int(tol_seconds * FPS)
    hits = sum(1 for p in peaks if bounds.size and np.min(np.abs(bounds - p)) <= tol)
    return float(hits / peaks.size)


def seam_jerk(motion: np.ndarray, sections: list, window: int = 15) -> tuple[float, float]:
    """FlowMDM-style transition metrics around section seams: (peak_jerk, area_under_jerk).

    Jerk = |3rd derivative| of the kinematic channels. Measured in a +/-``window`` band around each
    interior section boundary. Lower is smoother.
    """
    L = motion.shape[0]
    if L < 8 or len(sections) < 2:
        return 0.0, 0.0
    jerk = np.linalg.norm(np.diff(motion[:, :_KIN], n=3, axis=0), axis=1)
    peak, area, count = 0.0, 0.0, 0
    for s in sections[1:]:
        c = s.start_frame
        lo, hi = max(0, c - window), min(jerk.shape[0], c + window)
        if hi <= lo:
            continue
        band = jerk[lo:hi]
        peak = max(peak, float(band.max()))
        area += float(band.sum())
        count += 1
    return float(peak), float(area / max(count, 1))


def section_repetition_correlation(motion: np.ndarray, sections: list) -> float:
    """Raw dynamic similarity between same-label sections before transformation invariance.

    This preserves a strict view of literal repetition, while :func:`motif_recurrence` separately
    recognizes mirrored, retrograded, and phase-shifted choreographic variations.
    """
    segments = [
        motion[s.start_frame:s.end_frame]
        for s in sections
        if s.end_frame > s.start_frame
    ]
    labels = [s.label for s in sections if s.end_frame > s.start_frame]
    sims = []
    for i in range(len(segments)):
        for j in range(i + 1, len(segments)):
            if labels[i] == labels[j]:
                sims.append(_cosine_motion_similarity(segments[i], segments[j]))
    return float(np.mean(sims)) if sims else 0.0


def compute_story_metrics(motion: np.ndarray, structure, *, music_beat_frames=None) -> dict:
    """Aggregate all structure metrics for an assembled dance + its MusicStructure.

    When ``music_beat_frames`` (motion-frame indices, e.g. ``metadata.beat_frames``) are provided,
    beat-alignment metrics (BAS, coverage, foot-contact consistency) are included too.
    """
    sections = getattr(structure, "sections", [])
    peak_jerk, auj = seam_jerk(motion, sections)
    metrics = {
        "arc_adherence": round(arc_adherence(motion, getattr(structure, "energy_curve",
                                                             np.zeros(0))), 4),
        "sectional_contrast": round(sectional_contrast(motion, sections), 4),
        "motif_recurrence": round(motif_recurrence(motion, sections), 4),
        "section_repetition_correlation": round(section_repetition_correlation(motion, sections), 4),
        "boundary_alignment": round(boundary_alignment(motion, sections), 4),
        "peak_jerk": round(peak_jerk, 4),
        "area_under_jerk": round(auj, 4),
    }
    if music_beat_frames is not None:
        from agentlodge.dance.beat_metrics import compute_beat_metrics
        metrics.update(compute_beat_metrics(motion, music_beat_frames))
    return metrics
