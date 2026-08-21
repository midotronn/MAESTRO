#!/usr/bin/env python3
"""Run the authenticated HTTP task and artifact coordinator."""

from __future__ import annotations

import argparse
import os
import ssl
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from server.distributed.http_transport import (  # noqa: E402
    HttpCoordinatorStore,
    create_http_server,
)


def _token(args) -> str:
    token = str(os.environ.get("AGENTLODGE_HTTP_TOKEN", "")).strip()
    token_file = str(
        args.token_file or os.environ.get("AGENTLODGE_HTTP_TOKEN_FILE", "")
    ).strip()
    if not token and token_file:
        token = Path(token_file).read_text(encoding="utf-8").strip()
    if not token:
        raise SystemExit(
            "set AGENTLODGE_HTTP_TOKEN or AGENTLODGE_HTTP_TOKEN_FILE"
        )
    return token


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bind",
        default=os.environ.get("AGENTLODGE_HTTP_BIND", "127.0.0.1"),
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("AGENTLODGE_HTTP_PORT", "8765")),
    )
    parser.add_argument(
        "--state-root",
        default=os.environ.get("AGENTLODGE_HTTP_STATE_ROOT"),
    )
    parser.add_argument(
        "--artifact-root",
        default=os.environ.get("AGENTLODGE_HTTP_ARTIFACT_ROOT"),
    )
    parser.add_argument("--token-file")
    parser.add_argument("--tls-cert")
    parser.add_argument("--tls-key")
    parser.add_argument(
        "--lease-seconds",
        type=float,
        default=float(os.environ.get("AGENTLODGE_HTTP_LEASE_SECONDS", "30")),
    )
    parser.add_argument(
        "--worker-max-age",
        type=float,
        default=float(
            os.environ.get("AGENTLODGE_WORKER_HEARTBEAT_MAX_AGE", "30")
        ),
    )
    parser.add_argument(
        "--max-artifact-bytes",
        type=int,
        default=int(
            os.environ.get(
                "AGENTLODGE_HTTP_MAX_ARTIFACT_BYTES",
                str(16 * 1024 * 1024 * 1024),
            )
        ),
    )
    args = parser.parse_args()
    if not args.state_root:
        parser.error(
            "--state-root or AGENTLODGE_HTTP_STATE_ROOT is required"
        )
    if bool(args.tls_cert) != bool(args.tls_key):
        parser.error("--tls-cert and --tls-key must be provided together")

    store = HttpCoordinatorStore(
        Path(args.state_root),
        Path(args.artifact_root) if args.artifact_root else None,
        default_lease_seconds=args.lease_seconds,
        worker_max_age=args.worker_max_age,
        max_artifact_bytes=args.max_artifact_bytes,
    )
    server = create_http_server(
        args.bind,
        args.port,
        token=_token(args),
        store=store,
    )
    if args.tls_cert:
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        context.load_cert_chain(args.tls_cert, args.tls_key)
        server.socket = context.wrap_socket(server.socket, server_side=True)
    scheme = "https" if args.tls_cert else "http"
    host, port = server.server_address[:2]
    print(f"AgentLODGE HTTP coordinator listening on {scheme}://{host}:{port}")
    try:
        server.serve_forever(poll_interval=0.2)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
