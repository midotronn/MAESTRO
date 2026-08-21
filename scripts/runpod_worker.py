#!/usr/bin/env python3
"""Run one persistent capability worker over filesystem or authenticated HTTP."""

from __future__ import annotations

import argparse
import os
import signal
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from server.distributed.handlers import (  # noqa: E402
    AudioPreprocessHandler,
    BeatArtifactHandler,
    DanceGenerationHandler,
    EdgeGenerateHandler,
    JukeboxExtractHandler,
    LodgeGenerateHandler,
    RenderFramesHandler,
)
from server.distributed.http_transport import (  # noqa: E402
    HttpTaskWorker,
    HttpTransportError,
)
from server.distributed.render_contract import render_identity_digest  # noqa: E402
from server.distributed.registry import WorkerSpec  # noqa: E402
from server.distributed.worker import FileTaskWorker  # noqa: E402


def _http_token(args) -> str:
    token = str(args.auth_token or os.environ.get("AGENTLODGE_HTTP_TOKEN", "")).strip()
    token_file = str(
        args.auth_token_file
        or os.environ.get("AGENTLODGE_HTTP_TOKEN_FILE", "")
    ).strip()
    if not token and token_file:
        try:
            token = Path(token_file).read_text(encoding="utf-8").strip()
        except OSError as exc:
            raise HttpTransportError(
                f"could not read HTTP transport token file: {token_file}"
            ) from exc
    if not token:
        raise HttpTransportError("HTTP transport authentication token is required")
    return token


def _release_shm_reservation() -> None:
    configured = os.environ.get("AGENTLODGE_SHM_RESERVATION_FILE", "").strip()
    if not configured:
        return
    shm_root = Path(
        os.environ.get("AGENTLODGE_SHM_ROOT", "/dev/shm")
    ).resolve()
    reservation_root = (shm_root / ".agentlodge-reservations").resolve()
    reservation = Path(configured).resolve()
    if not reservation.is_relative_to(reservation_root):
        return
    try:
        owner_pid = int(
            reservation.read_text(encoding="utf-8").split(maxsplit=1)[0]
        )
    except (OSError, ValueError, IndexError):
        return
    if owner_pid == os.getpid():
        reservation.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker-id", required=True)
    parser.add_argument(
        "--transport",
        choices=["filesystem", "http"],
        default=os.environ.get(
            "AGENTLODGE_DISTRIBUTED_TRANSPORT",
            "filesystem",
        ),
    )
    parser.add_argument("--task-dir")
    parser.add_argument("--shared-root")
    parser.add_argument(
        "--coordinator-url",
        default=os.environ.get("AGENTLODGE_HTTP_COORDINATOR_URL"),
    )
    parser.add_argument("--auth-token")
    parser.add_argument("--auth-token-file")
    parser.add_argument("--worker-scratch")
    parser.add_argument(
        "--workspace-root",
        default=os.environ.get("WORKSPACE", "/workspace"),
    )
    parser.add_argument(
        "--lease-seconds",
        type=float,
        default=float(os.environ.get("AGENTLODGE_HTTP_LEASE_SECONDS", "30")),
    )
    parser.add_argument(
        "--capability",
        required=True,
        choices=[
            "jukebox.extract",
            "audio.lodge",
            "audio.edge",
            "audio.beats",
            "dance.generate",
            "lodge.generate",
            "edge.generate",
            "render.frames",
        ],
    )
    parser.add_argument("--edge-root")
    parser.add_argument("--edge-checkpoint")
    parser.add_argument("--lodge-root")
    parser.add_argument("--lodge-weights")
    parser.add_argument("--lodge-global-weights")
    parser.add_argument("--lodge-genre", default="Hiphop")
    parser.add_argument(
        "--render-width",
        type=int,
        default=int(os.environ.get("AGENTLODGE_RENDER_FULL_W", "1080")),
    )
    parser.add_argument(
        "--render-height",
        type=int,
        default=int(os.environ.get("AGENTLODGE_RENDER_FULL_H", "1080")),
    )
    parser.add_argument(
        "--render-samples",
        type=int,
        default=int(
            os.environ.get("AGENTLODGE_RENDER_FULL_SAMPLES", "96")
        ),
    )
    parser.add_argument(
        "--render-engine",
        default=os.environ.get("AGENTLODGE_RENDER_ENGINE", "eevee"),
    )
    parser.add_argument(
        "--render-denoise",
        type=int,
        default=int(os.environ.get("AGENTLODGE_RENDER_DENOISE", "1")),
    )
    parser.add_argument(
        "--render-frame-format",
        default=os.environ.get("AGENTLODGE_RENDER_FRAME_FORMAT", "tga"),
    )
    parser.add_argument("--worker-tmp")
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=float(os.environ.get("AGENTLODGE_WORKER_POLL_INTERVAL", "0.1")),
    )
    parser.add_argument(
        "--heartbeat-interval",
        type=float,
        default=float(
            os.environ.get("AGENTLODGE_WORKER_HEARTBEAT_INTERVAL", "2.0")
        ),
    )
    parser.add_argument(
        "--no-preload",
        action="store_true",
        help="Defer model loading until the first task.",
    )
    args = parser.parse_args()
    render_provenance = None
    render_quality = None
    render_identity = None
    if args.transport == "http" and args.capability != "render.frames":
        parser.error("HTTP transport is currently supported only for render.frames")
    if args.transport == "filesystem":
        if not args.task_dir:
            parser.error("--task-dir is required for filesystem transport")
        if not args.shared_root:
            parser.error("--shared-root is required for filesystem transport")
        shared_root = Path(args.shared_root).resolve()
    else:
        if not args.coordinator_url:
            parser.error(
                "--coordinator-url or AGENTLODGE_HTTP_COORDINATOR_URL is required"
            )
        scratch_value = (
            args.worker_scratch
            or args.worker_tmp
            or os.environ.get("AGENTLODGE_HTTP_WORKER_SCRATCH")
        )
        if not scratch_value:
            parser.error(
                "--worker-scratch or AGENTLODGE_HTTP_WORKER_SCRATCH is required"
            )
        shared_root = Path(scratch_value).resolve()

    if args.capability == "jukebox.extract":
        if not args.edge_root:
            parser.error("--edge-root is required for jukebox.extract")
        jukebox_scratch = os.environ.get(
            "AGENTLODGE_JUKEBOX_SHARED_SCRATCH", ""
        ).strip()
        handler = JukeboxExtractHandler(
            edge_root=Path(args.edge_root),
            shared_root=shared_root,
            scratch_root=Path(jukebox_scratch) if jukebox_scratch else None,
        )
    elif args.capability == "audio.lodge":
        if not args.lodge_root:
            parser.error("--lodge-root is required for audio.lodge")
        handler = AudioPreprocessHandler(
            mode="lodge",
            shared_root=shared_root,
            lodge_root=Path(args.lodge_root),
        )
    elif args.capability == "audio.edge":
        if not args.edge_root:
            parser.error("--edge-root is required for audio.edge")
        jukebox_scratch = os.environ.get(
            "AGENTLODGE_JUKEBOX_SHARED_SCRATCH", ""
        ).strip()
        handler = AudioPreprocessHandler(
            mode="edge",
            shared_root=shared_root,
            edge_root=Path(args.edge_root),
            scratch_root=Path(jukebox_scratch) if jukebox_scratch else None,
        )
    elif args.capability == "audio.beats":
        handler = BeatArtifactHandler(shared_root=shared_root)
    elif args.capability == "dance.generate":
        handler = DanceGenerationHandler(shared_root=shared_root)
    elif args.capability == "lodge.generate":
        required = {
            "--lodge-root": args.lodge_root,
            "--lodge-weights": args.lodge_weights,
            "--lodge-global-weights": args.lodge_global_weights,
        }
        missing = [name for name, value in required.items() if not value]
        if missing:
            parser.error(f"{', '.join(missing)} required for lodge.generate")
        handler = LodgeGenerateHandler(
            shared_root=shared_root,
            lodge_root=Path(args.lodge_root),
            lodge_weights=Path(args.lodge_weights),
            lodge_global_weights=Path(args.lodge_global_weights),
            genre=args.lodge_genre,
        )
    elif args.capability == "edge.generate":
        if not args.edge_root or not args.edge_checkpoint:
            parser.error(
                "--edge-root and --edge-checkpoint are required for edge.generate"
            )
        handler = EdgeGenerateHandler(
            shared_root=shared_root,
            edge_root=Path(args.edge_root),
            checkpoint=Path(args.edge_checkpoint),
        )
    elif args.capability == "render.frames":
        os.environ.setdefault("AGENTLODGE_WARM_POOL", "1")
        os.environ.setdefault("AGENTLODGE_POD_HOST", "127.0.0.1")
        os.environ.setdefault(
            "AGENTLODGE_POD_WS",
            str(
                shared_root
                if args.transport == "filesystem"
                else Path(args.workspace_root).resolve()
            ),
        )
        handler = RenderFramesHandler(
            shared_root=shared_root,
            width=args.render_width,
            height=args.render_height,
            samples=args.render_samples,
            engine=args.render_engine,
            denoise=args.render_denoise,
            frame_format=args.render_frame_format,
            local_tmp=(
                Path(args.worker_tmp).resolve()
                if args.worker_tmp
                else shared_root
            ),
            worker_id=args.worker_id,
        )
        render_provenance = handler.render_provenance()
        render_quality = {
            "width": args.render_width,
            "height": args.render_height,
            "samples": args.render_samples,
            "engine": args.render_engine,
            "denoise": args.render_denoise,
            "frame_format": args.render_frame_format,
            "fps": 30,
        }
        render_identity = render_identity_digest(
            render_provenance,
            render_quality,
        )
    else:
        parser.error(f"unsupported capability: {args.capability}")

    if args.transport == "filesystem":
        spec = WorkerSpec(
            worker_id=args.worker_id,
            capabilities=(args.capability,),
            task_dir=Path(args.task_dir),
            metadata={
                "shared_root": str(shared_root),
                "gpu_index": os.environ.get(
                    "AGENTLODGE_RESOLVED_GPU_INDEX",
                    os.environ.get("AGENTLODGE_GPU_INDEX", ""),
                ),
                "render_daemon_root": os.environ.get(
                    "AGENTLODGE_RENDER_DAEMON_ROOT",
                    "",
                ),
                **(
                    {
                        "render_provenance": render_provenance,
                        "quality": {
                            key: render_quality[key]
                            for key in (
                                "width",
                                "height",
                                "samples",
                                "engine",
                                "denoise",
                                "frame_format",
                            )
                        },
                        "render_identity_digest": render_identity,
                    }
                    if render_provenance is not None
                    else {}
                ),
            },
        )
        worker = FileTaskWorker(
            spec,
            {args.capability: handler},
            poll_interval=args.poll_interval,
            heartbeat_interval=args.heartbeat_interval,
        )
    else:
        worker = HttpTaskWorker(
            args.worker_id,
            (args.capability,),
            {args.capability: handler},
            base_url=args.coordinator_url,
            token=_http_token(args),
            scratch_root=shared_root,
            metadata={
                "transport": "http",
                "scratch_root": str(shared_root),
                "gpu_index": os.environ.get(
                    "AGENTLODGE_RESOLVED_GPU_INDEX",
                    os.environ.get("AGENTLODGE_GPU_INDEX", ""),
                ),
                "render_daemon_root": os.environ.get(
                    "AGENTLODGE_RENDER_DAEMON_ROOT",
                    "",
                ),
                **(
                    {
                        "render_provenance": render_provenance,
                        "quality": {
                            key: render_quality[key]
                            for key in (
                                "width",
                                "height",
                                "samples",
                                "engine",
                                "denoise",
                                "frame_format",
                            )
                        },
                        "render_identity_digest": render_identity,
                    }
                    if render_provenance is not None
                    else {}
                ),
            },
            poll_interval=args.poll_interval,
            heartbeat_interval=args.heartbeat_interval,
            lease_seconds=args.lease_seconds,
            request_timeout=float(
                os.environ.get("AGENTLODGE_HTTP_REQUEST_TIMEOUT", "30")
            ),
        )

    def stop(_signum, _frame):
        worker.stop()

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)

    try:
        if not args.no_preload:
            worker.set_status("warming")
            worker.start_heartbeat()
            handler.preload()
        worker.run_forever()
    finally:
        if args.capability == "render.frames":
            _release_shm_reservation()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
