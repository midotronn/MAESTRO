#!/usr/bin/env python3
"""Run one persistent, capability-scoped worker against a shared task directory."""

from __future__ import annotations

import argparse
import os
import signal
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from server.distributed.handlers import (  # noqa: E402
    EdgeGenerateHandler,
    JukeboxExtractHandler,
    LodgeGenerateHandler,
    RenderFramesHandler,
)
from server.distributed.registry import WorkerSpec  # noqa: E402
from server.distributed.worker import FileTaskWorker  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker-id", required=True)
    parser.add_argument("--task-dir", required=True)
    parser.add_argument("--shared-root", required=True)
    parser.add_argument(
        "--capability",
        required=True,
        choices=[
            "jukebox.extract",
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
    parser.add_argument("--render-width", type=int, default=1080)
    parser.add_argument("--render-height", type=int, default=1080)
    parser.add_argument("--render-samples", type=int, default=96)
    parser.add_argument("--render-engine", default="eevee")
    parser.add_argument("--render-denoise", type=int, default=1)
    parser.add_argument("--render-frame-format", default="tga")
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

    if args.capability == "jukebox.extract":
        if not args.edge_root:
            parser.error("--edge-root is required for jukebox.extract")
        handler = JukeboxExtractHandler(
            edge_root=Path(args.edge_root),
            shared_root=Path(args.shared_root),
        )
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
            shared_root=Path(args.shared_root),
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
            shared_root=Path(args.shared_root),
            edge_root=Path(args.edge_root),
            checkpoint=Path(args.edge_checkpoint),
        )
    elif args.capability == "render.frames":
        os.environ.setdefault("AGENTLODGE_WARM_POOL", "1")
        os.environ.setdefault("AGENTLODGE_POD_HOST", "127.0.0.1")
        os.environ.setdefault("AGENTLODGE_POD_WS", str(Path(args.shared_root)))
        handler = RenderFramesHandler(
            shared_root=Path(args.shared_root),
            width=args.render_width,
            height=args.render_height,
            samples=args.render_samples,
            engine=args.render_engine,
            denoise=args.render_denoise,
            frame_format=args.render_frame_format,
            local_tmp=Path(args.worker_tmp) if args.worker_tmp else None,
        )
    else:
        parser.error(f"unsupported capability: {args.capability}")

    spec = WorkerSpec(
        worker_id=args.worker_id,
        capabilities=(args.capability,),
        task_dir=Path(args.task_dir),
        metadata={"shared_root": str(Path(args.shared_root).resolve())},
    )
    worker = FileTaskWorker(
        spec,
        {args.capability: handler},
        poll_interval=args.poll_interval,
        heartbeat_interval=args.heartbeat_interval,
    )

    def stop(_signum, _frame):
        worker.stop()

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)

    if not args.no_preload:
        worker.set_status("warming")
        worker.start_heartbeat()
        handler.preload()
    worker.run_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
