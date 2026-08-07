"""End-to-end AgentLODGE pipeline orchestration."""

from __future__ import annotations

import json
import logging
import os
import tempfile
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from agentlodge.agent.costume_describer import describe_costume
from agentlodge.agent.selector import SelectionResult, select_dance
from agentlodge.audio.preprocess import PreprocessedAudio, preprocess_audio, release_torch_memory
from agentlodge.config import FPS, Settings
from agentlodge.dance.format import to_editor139
from agentlodge.dance.best_of_k import best_of_k_job as _best_of_k_job
from agentlodge.dance.hybrid import build_hybrid, _merge_runs
from agentlodge.audio.structure import analyze_structure
from agentlodge.agent.storyboard import author_storyboard
from agentlodge.dance.story import build_story_dance
from agentlodge.dance.story_metrics import compute_story_metrics
from agentlodge.dance.metrics import DanceMetrics, compute_metrics
from agentlodge.image.costume import generate_costume_image
from agentlodge.subprocess_runner import (
    run_blender_video_subprocess,
    run_edge_inference_subprocess,
    run_lodge_inference_subprocess,
    run_stick_video_subprocess,
    run_ybot_video_subprocess,
)

logger = logging.getLogger(__name__)


@dataclass
class GenerationOutcome:
    motion: np.ndarray | None = None
    summary: str = ""
    error: str | None = None


@dataclass
class PipelineResult:
    output_dir: Path
    selected_model: str
    selection_reasoning: str
    lodge_metrics: DanceMetrics | None = None
    edge_metrics: DanceMetrics | None = None
    song_duration_seconds: float = 0.0
    costume_description: str = ""
    stick_figure_video: Path | None = None
    storyboard: dict | None = None
    structure_metrics: dict | None = None
    errors: list[str] = field(default_factory=list)


def _run_lodge_job(
    lodge_features: np.ndarray,
    settings_dict: dict,
    work_dir: str,
    seed: int | None = None,
) -> dict:
    settings = Settings.from_dict(settings_dict)
    work = Path(work_dir).resolve()
    work.mkdir(parents=True, exist_ok=True)
    features_path = (work / "lodge_features.npy").resolve()
    output_path = (work / "lodge_motion.npy").resolve()
    np.save(features_path, lodge_features)
    try:
        summary = run_lodge_inference_subprocess(
            settings.lodge_code_path.resolve(),
            settings.lodge_weights_path.resolve(),
            settings.lodge_global_weights_path.resolve(),
            settings.lodge_genre,
            features_path,
            output_path,
            work,
            seed=seed,
        )
        motion = np.load(output_path)
        return {"motion": motion, "summary": summary, "error": None}
    except Exception as exc:
        return {
            "motion": None,
            "summary": "",
            "error": f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}",
        }


def _run_edge_job(
    wav_path: str,
    edge_slices: list,
    settings_dict: dict,
    work_dir: str,
    seed: int | None = None,
) -> dict:
    settings = Settings.from_dict(settings_dict)
    work = Path(work_dir).resolve()
    work.mkdir(parents=True, exist_ok=True)
    features_path = (work / "edge_features.npy").resolve()
    output_path = (work / "edge_motion.npy").resolve()
    np.save(features_path, np.array(edge_slices))

    num_clips = len(edge_slices)
    overlap_seconds = 2.5
    clip_seconds = 5.0
    expected_frames = int(
        clip_seconds * FPS + (num_clips - 1) * overlap_seconds * FPS
    )
    try:
        run_edge_inference_subprocess(
            settings.edge_code_path.resolve(),
            settings.edge_weights_path.resolve(),
            work,
            features_path,
            output_path,
            seed=seed,
        )
        motion = np.load(output_path).astype(np.float32)
        if motion.shape[0] > expected_frames:
            motion = motion[:expected_frames]
        summary = (
            f"EDGE long-form pipeline with {num_clips} chained 5s clips "
            f"and 2.5s overlap; output length {motion.shape[0]} frames."
        )
        return {"motion": motion, "summary": summary, "error": None}
    except Exception as exc:
        return {
            "motion": None,
            "summary": "",
            "error": f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}",
        }


def _use_parallel_execution() -> bool:
    mode = os.getenv("AGENTLODGE_PARALLEL", "auto").lower()
    if mode in {"0", "false", "no"}:
        return False
    if mode in {"1", "true", "yes"}:
        return True
    try:
        import torch

        return torch.cuda.is_available()
    except ImportError:
        return False


def _run_generators(
    preprocessed: PreprocessedAudio,
    settings: Settings,
    settings_dict: dict,
    work_root: Path,
) -> tuple[GenerationOutcome, GenerationOutcome]:
    lodge_out = GenerationOutcome()
    edge_out = GenerationOutcome()

    if _use_parallel_execution():
        logger.info("Running LODGE and EDGE generation in parallel...")
        with ProcessPoolExecutor(max_workers=2) as executor:
            futures = {
                executor.submit(
                    _run_lodge_job,
                    preprocessed.lodge_features,
                    settings_dict,
                    str(work_root / "lodge"),
                ): "lodge",
                executor.submit(
                    _run_edge_job,
                    str(preprocessed.wav_path),
                    preprocessed.edge_feature_slices or [],
                    settings_dict,
                    str(work_root),
                ): "edge",
            }
            for future in as_completed(futures):
                name = futures[future]
                payload = future.result()
                outcome = lodge_out if name == "lodge" else edge_out
                outcome.motion = payload["motion"]
                outcome.summary = payload["summary"]
                outcome.error = payload["error"]
    else:
        logger.info("Running LODGE and EDGE generation sequentially...")
        for name, runner in (
            ("lodge", lambda: _run_lodge_job(
                preprocessed.lodge_features,
                settings_dict,
                str(work_root / "lodge"),
            )),
            ("edge", lambda: _run_edge_job(
                str(preprocessed.wav_path),
                preprocessed.edge_feature_slices or [],
                settings_dict,
                str(work_root),
            )),
        ):
            payload = runner()
            outcome = lodge_out if name == "lodge" else edge_out
            outcome.motion = payload["motion"]
            outcome.summary = payload["summary"]
            outcome.error = payload["error"]

    return lodge_out, edge_out


def _settings_to_dict(settings: Settings) -> dict:
    return {
        "anthropic_api_key": settings.anthropic_api_key,
        "openai_api_key": settings.openai_api_key,
        "gemini_api_key": settings.gemini_api_key,
        "image_backend": settings.image_backend,
        "output_dir": str(settings.output_dir),
        "lodge_code_path": str(settings.lodge_code_path),
        "edge_code_path": str(settings.edge_code_path),
        "lodge_weights_path": str(settings.lodge_weights_path),
        "lodge_global_weights_path": str(settings.lodge_global_weights_path),
        "edge_weights_path": str(settings.edge_weights_path),
        "lodge_genre": settings.lodge_genre,
        "min_audio_seconds": settings.min_audio_seconds,
        "max_edge_slices": settings.max_edge_slices,
        "render_backend": settings.render_backend,
        "blender_bin": str(settings.blender_bin) if settings.blender_bin else None,
        "smplx_model_dir": str(settings.smplx_model_dir) if settings.smplx_model_dir else None,
        "smplx_uv_path": str(settings.smplx_uv_path) if settings.smplx_uv_path else None,
        "smplx_texture_path": str(settings.smplx_texture_path) if settings.smplx_texture_path else None,
        "hybrid_enabled": settings.hybrid_enabled,
        "hybrid_min_seg_seconds": settings.hybrid_min_seg_seconds,
        "hybrid_blend_frames": settings.hybrid_blend_frames,
        "hybrid_scheduler": settings.hybrid_scheduler,
        "hybrid_expressiveness": settings.hybrid_expressiveness,
        "hybrid_canonical_facing": settings.hybrid_canonical_facing,
        "story_enabled": settings.story_enabled,
        "story_motif_reuse": settings.story_motif_reuse,
        "story_energy_shaping": settings.story_energy_shaping,
        "story_min_section_seconds": settings.story_min_section_seconds,
        "story_recapitulate": settings.story_recapitulate,
        "structure_spectral": settings.structure_spectral,
        "best_of_k": settings.best_of_k,
    }


def _lodge_score_transform(motion):
    """Convert a raw LODGE motion to the AgentLODGE 139-dim Z-up scoring format (needs torch)."""
    from agentlodge.dance.format import ensure_lodge139, to_agentlodge139
    from agentlodge.dance.transition import to_zup
    return to_zup(to_agentlodge139(ensure_lodge139(motion)))


def _edge_score_transform(motion):
    """Convert a raw EDGE motion to the AgentLODGE 139-dim scoring format (EDGE already Z-up)."""
    from agentlodge.dance.format import ensure_lodge139, to_agentlodge139
    return to_agentlodge139(ensure_lodge139(motion))


def run_pipeline(
    audio_path: str | Path,
    output_dir: str | Path | None = None,
    settings: Settings | None = None,
    *,
    render_video: bool = True,
) -> PipelineResult:
    settings = settings or Settings.from_env()
    out_dir = Path(output_dir or settings.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    errors: list[str] = []
    work_root = Path(tempfile.mkdtemp(prefix="agentlodge_", dir=out_dir)).resolve()

    audio_path = Path(audio_path).resolve()
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    logger.info("Preprocessing audio (Lodge features)...")
    preprocessed = preprocess_audio(
        audio_path, settings, work_root, extract_edge=False
    )
    if preprocessed.metadata.duration_seconds < settings.min_audio_seconds:
        raise ValueError(
            f"Audio is {preprocessed.metadata.duration_seconds:.1f}s; "
            f"LODGE requires at least {settings.min_audio_seconds:.0f}s "
            "(two 256-frame local windows)."
        )
    if preprocessed.lodge_features.shape[0] // 256 < 2:
        raise ValueError(
            "Audio is too short for LODGE fine diffusion. "
            f"Need at least {settings.min_audio_seconds:.0f}s of music."
        )

    settings_dict = _settings_to_dict(settings)
    _music_beats = getattr(preprocessed.metadata, "beat_frames", None)
    lodge_out = _best_of_k_job(
        lambda seed: _run_lodge_job(
            preprocessed.lodge_features,
            settings_dict,
            str(work_root / "lodge" / (f"seed_{seed}" if seed is not None else "single")),
            seed=seed,
        ),
        settings.best_of_k, _music_beats, _lodge_score_transform,
    )
    lodge_result = GenerationOutcome(
        motion=lodge_out["motion"],
        summary=lodge_out["summary"],
        error=lodge_out["error"],
    )
    release_torch_memory()

    logger.info("Preprocessing audio (EDGE Jukebox features)...")
    try:
        edge_preprocessed = preprocess_audio(
            preprocessed.wav_path,
            settings,
            work_root,
            extract_lodge=False,
            extract_edge=True,
        )
        preprocessed.edge_feature_slices = edge_preprocessed.edge_feature_slices
    except Exception as exc:
        msg = f"EDGE preprocessing failed: {exc}"
        errors.append(msg)
        logger.error(msg)
        edge_preprocessed = None
    release_torch_memory()

    edge_result = GenerationOutcome()
    if preprocessed.edge_feature_slices:
        edge_out = _best_of_k_job(
            lambda seed: _run_edge_job(
                str(preprocessed.wav_path),
                preprocessed.edge_feature_slices,
                settings_dict,
                str(work_root),
                seed=seed,
            ),
            settings.best_of_k, _music_beats, _edge_score_transform,
        )
        edge_result = GenerationOutcome(
            motion=edge_out["motion"],
            summary=edge_out["summary"],
            error=edge_out["error"],
        )
        release_torch_memory()

    lodge_out = lodge_result
    edge_out = edge_result
    for name, outcome in (("lodge", lodge_out), ("edge", edge_out)):
        if outcome.error:
            errors.append(f"{name} generation failed: {outcome.error}")
            logger.error("%s generation failed: %s", name, outcome.error)

    if lodge_out.motion is None and edge_out.motion is None:
        raise RuntimeError(
            "Both LODGE and EDGE generation failed.\n" + "\n".join(errors)
        )

    lodge_metrics = None
    edge_metrics = None
    if lodge_out.motion is not None:
        lodge_metrics = compute_metrics(
            lodge_out.motion,
            preprocessed.metadata,
            "lodge",
            lodge_out.summary,
        )
    if edge_out.motion is not None:
        edge_metrics = compute_metrics(
            edge_out.motion,
            preprocessed.metadata,
            "edge",
            edge_out.summary,
        )

    selected_model = "lodge"
    reasoning = ""
    selection_analysis = ""
    selection_scores: dict | None = None
    hybrid_schedule: list | None = None
    story_schedule: list | None = None
    story_section_scores: list | None = None
    structure_payload: dict | None = None
    storyboard_payload: dict | None = None
    structure_metrics_payload: dict | None = None
    selected_motion = None

    if (
        settings.story_enabled
        and lodge_out.motion is not None
        and edge_out.motion is not None
    ):
        try:
            total_frames = int(min(lodge_out.motion.shape[0], edge_out.motion.shape[0]))
            structure = analyze_structure(
                preprocessed.wav_path,
                preprocessed.metadata,
                total_frames,
                min_section_seconds=settings.story_min_section_seconds,
                spectral=settings.structure_spectral,
            )
            storyboard = author_storyboard(
                structure,
                preprocessed.metadata,
                preprocessed.audio_descriptor,
                settings.openai_api_key,
                motif_reuse=settings.story_motif_reuse,
            )
            story = build_story_dance(
                lodge_out.motion,
                edge_out.motion,
                structure,
                storyboard,
                preprocessed.metadata,
                blend_frames=settings.hybrid_blend_frames,
                motif_reuse=settings.story_motif_reuse,
                energy_shaping=settings.story_energy_shaping,
                recapitulate=settings.story_recapitulate,
            )
            selected_model = "story"
            selected_motion = story.motion
            reasoning = story.reasoning
            story_schedule = [
                [int(a), int(b), str(src), str(role)] for a, b, src, role in story.schedule
            ]
            structure_payload = structure.to_dict()
            storyboard_payload = storyboard.to_dict()
            story_section_scores = story.section_scores
            structure_metrics_payload = compute_story_metrics(
                story.motion, structure,
                music_beat_frames=getattr(preprocessed.metadata, "beat_frames", None),
            )
            logger.info("Story dance assembled (%d sections): %s",
                        len(story_schedule), story.reasoning)
        except Exception as exc:
            msg = f"Story assembly failed: {exc}; falling back to hybrid/single-model."
            errors.append(msg)
            logger.error(msg)

    if (
        selected_motion is None
        and settings.hybrid_enabled
        and lodge_out.motion is not None
        and edge_out.motion is not None
    ):
        try:
            hybrid = build_hybrid(
                lodge_out.motion,
                edge_out.motion,
                preprocessed.metadata,
                min_seg_seconds=settings.hybrid_min_seg_seconds,
                blend_frames=settings.hybrid_blend_frames,
                scheduler=settings.hybrid_scheduler,
                api_key=settings.openai_api_key,
                expressiveness=settings.hybrid_expressiveness,
                canonical_facing=settings.hybrid_canonical_facing,
                lodge_code_path=settings.lodge_code_path,
            )
            selected_model = "hybrid"
            selected_motion = hybrid.motion
            reasoning = hybrid.reasoning
            hybrid_schedule = [
                [int(a), int(b), g] for a, b, g in _merge_runs(hybrid.schedule)
            ]
            logger.info("Hybrid dance assembled (%d runs): %s",
                        len(hybrid_schedule), hybrid_schedule)
        except Exception as exc:
            msg = f"Hybrid assembly failed: {exc}; falling back to single-model selection."
            errors.append(msg)
            logger.error(msg)

    if selected_motion is None:
        if lodge_metrics and edge_metrics:
            selection = select_dance(
                lodge_metrics,
                edge_metrics,
                preprocessed.metadata,
                settings.openai_api_key,
            )
            selected_model = selection.selected_model
            reasoning = selection.reasoning
            selection_analysis = selection.analysis
            selection_scores = selection.scores
            if selection.used_fallback:
                errors.append(f"selection agent fallback: {reasoning}")
        elif lodge_out.motion is None:
            selected_model = "edge"
            reasoning = "LODGE failed; automatically selected EDGE output."
        elif edge_out.motion is None:
            selected_model = "lodge"
            reasoning = "EDGE failed; automatically selected LODGE output."

        selected_motion = lodge_out.motion if selected_model == "lodge" else edge_out.motion
        if selected_motion is None:
            selected_motion = edge_out.motion if lodge_out.motion is None else lodge_out.motion
            selected_model = "edge" if lodge_out.motion is None else "lodge"

    # Persist one canonical format no matter which backbone won. The renderer can auto-detect raw
    # LODGE contact-first/Y-up arrays, but the editor cannot compose a Z-up named motion into them.
    selected_139 = to_editor139(selected_motion)
    motion_path = out_dir / "selected_dance.npy"
    np.save(motion_path, selected_139)

    stick_figure_video: Path | None = None
    if render_video:
        want_ybot = (
            settings.render_backend == "blender"
            and settings.render_character == "ybot"
        )
        use_ybot = (
            want_ybot
            and settings.blender_bin is not None
            and settings.ybot_fbx_path is not None
        )
        use_blender = (
            settings.render_backend == "blender"
            and not want_ybot
            and settings.blender_bin is not None
            and settings.smplx_model_dir is not None
        )
        if want_ybot and not use_ybot:
            msg = (
                "render_character=ybot but BLENDER_BIN/YBOT_FBX_PATH are not both set; "
                "falling back to stick-figure rendering."
            )
            errors.append(msg)
            logger.warning(msg)
        elif settings.render_backend == "blender" and not use_blender and not want_ybot:
            msg = (
                "render_backend=blender but BLENDER_BIN/SMPLX_MODEL_DIR are not both set; "
                "falling back to stick-figure rendering."
            )
            errors.append(msg)
            logger.warning(msg)

        if use_ybot:
            stick_figure_video = out_dir / "dance_blender.mp4"
            try:
                run_ybot_video_subprocess(
                    settings.lodge_code_path.resolve(),
                    motion_path,
                    stick_figure_video,
                    settings.blender_bin.resolve(),
                    settings.ybot_fbx_path.resolve(),
                    audio_path=audio_path,
                    color=settings.render_color,
                )
                logger.info("Saved Blender Y-Bot video to %s", stick_figure_video)
            except Exception as exc:
                msg = f"Blender Y-Bot video render failed: {exc}"
                errors.append(msg)
                logger.error(msg)
                stick_figure_video = None  # fall through to stick figure

        if use_blender:
            stick_figure_video = out_dir / "dance_blender.mp4"
            try:
                run_blender_video_subprocess(
                    settings.lodge_code_path.resolve(),
                    motion_path,
                    stick_figure_video,
                    settings.blender_bin.resolve(),
                    settings.smplx_model_dir.resolve(),
                    audio_path=audio_path,
                    uv_npz=settings.smplx_uv_path,
                    texture_png=settings.smplx_texture_path,
                )
                logger.info("Saved Blender mesh video to %s", stick_figure_video)
            except Exception as exc:
                msg = f"Blender mesh video render failed: {exc}"
                errors.append(msg)
                logger.error(msg)
                stick_figure_video = None
                use_blender = False  # fall through to stick figure

        if stick_figure_video is None:
            stick_figure_video = out_dir / "dance_stick_figure.mp4"
            try:
                run_stick_video_subprocess(
                    settings.lodge_code_path.resolve(),
                    motion_path,
                    stick_figure_video,
                    audio_path=audio_path,
                )
                logger.info("Saved stick figure video to %s", stick_figure_video)
            except Exception as exc:
                msg = f"Stick figure video render failed: {exc}"
                errors.append(msg)
                logger.error(msg)
                stick_figure_video = None

    costume_description = ""
    costume_path = out_dir / "costume_output.png"
    try:
        if preprocessed.audio_descriptor is None:
            raise RuntimeError("Audio descriptor unavailable for costume generation")
        costume_description = describe_costume(
            preprocessed.audio_descriptor,
            settings.openai_api_key,
            genre=settings.lodge_genre,
        )
        logger.info("Audio-derived costume description: %s", costume_description)
        generate_costume_image(costume_description, costume_path, settings)
        logger.info("Saved costume image to %s", costume_path)
    except Exception as exc:
        msg = f"Costume image generation failed: {exc}"
        errors.append(msg)
        logger.error(msg)

    descriptor = preprocessed.audio_descriptor
    log_payload = {
        "selected_model": selected_model,
        "selection_reasoning": reasoning,
        "selection_analysis": selection_analysis,
        "selection_scores": selection_scores,
        "hybrid_schedule": hybrid_schedule,
        "story_schedule": story_schedule,
        "story_section_scores": story_section_scores,
        "structure": structure_payload,
        "storyboard": storyboard_payload,
        "structure_metrics": structure_metrics_payload,
        "lodge_bas": lodge_metrics.beat_alignment_score if lodge_metrics else None,
        "edge_bas": edge_metrics.beat_alignment_score if edge_metrics else None,
        "lodge_diversity": lodge_metrics.motion_diversity if lodge_metrics else None,
        "edge_diversity": edge_metrics.motion_diversity if edge_metrics else None,
        "lodge_coherence": lodge_metrics.coherence.to_dict()
        if lodge_metrics and lodge_metrics.coherence
        else None,
        "edge_coherence": edge_metrics.coherence.to_dict()
        if edge_metrics and edge_metrics.coherence
        else None,
        "song_duration_seconds": preprocessed.metadata.duration_seconds,
        "costume_description": costume_description,
        "audio_features": {
            "bpm": descriptor.bpm,
            "tempo_feel": descriptor.tempo_feel,
            "energy_level": descriptor.energy_level,
            "brightness": descriptor.brightness,
            "rhythmic_density": descriptor.rhythmic_density,
            "key": f"{descriptor.key} {descriptor.mode}",
            "mood": descriptor.mood,
        }
        if descriptor
        else None,
        "stick_figure_video": str(stick_figure_video) if stick_figure_video else None,
        "errors": errors,
    }
    (out_dir / "pipeline_log.json").write_text(json.dumps(log_payload, indent=2))

    return PipelineResult(
        output_dir=out_dir,
        selected_model=selected_model,
        selection_reasoning=reasoning,
        lodge_metrics=lodge_metrics,
        edge_metrics=edge_metrics,
        song_duration_seconds=preprocessed.metadata.duration_seconds,
        costume_description=costume_description,
        stick_figure_video=stick_figure_video,
        storyboard=storyboard_payload,
        structure_metrics=structure_metrics_payload,
        errors=errors,
    )
