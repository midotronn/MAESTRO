"""Burn action labels into the rendered motion-bank reel.

The reel is one continuous take of every canonical action, so without
labels you cannot tell which action you are looking at or where one ends and
the next begins. This overlays the current action name and a progress counter
for exactly the frame range that action occupies, driving ffmpeg from the same
sidecar the reel builder wrote.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

FONT_CANDIDATES = (
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    "C:/Windows/Fonts/segoeuib.ttf",
    "C:/Windows/Fonts/arialbd.ttf",
)


def find_font(explicit: str | None) -> str:
    if explicit:
        if not Path(explicit).is_file():
            raise SystemExit(f"font not found: {explicit}")
        return explicit
    for candidate in FONT_CANDIDATES:
        if Path(candidate).is_file():
            return candidate
    raise SystemExit(
        "no usable font found; pass --font, or install fonts-dejavu-core on the pod"
    )


def escape(text: str) -> str:
    """Quote text for use inside a single-quoted ffmpeg filter argument."""
    return text.replace("\\", "\\\\").replace("'", r"\'")


def build_filter(spans: list[dict], font: str, height: int) -> str:
    font_arg = escape(font)
    title_size = max(20, height // 22)
    sub_size = max(14, height // 40)
    total = len(spans)
    parts = []
    for index, span in enumerate(spans, start=1):
        window = f"between(n,{span['start']},{max(span['start'], span['end'] - 1)})"
        counter = f"{index}/{total}  {span['id']}"
        parts.append(
            f"drawtext=fontfile='{font_arg}':text='{escape(span['label'])}'"
            f":fontsize={title_size}:fontcolor=white:borderw=3:bordercolor=black@0.8"
            f":x=(w-text_w)/2:y=h*0.06:enable='{window}'"
        )
        parts.append(
            f"drawtext=fontfile='{font_arg}':text='{escape(counter)}'"
            f":fontsize={sub_size}:fontcolor=white@0.75:borderw=2:bordercolor=black@0.8"
            f":x=(w-text_w)/2:y=h*0.06+{title_size + 10}:enable='{window}'"
        )
    return ",".join(parts)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", type=Path, required=True, help="rendered reel mp4")
    ap.add_argument("--labels", type=Path, required=True, help="review_reel.labels.json")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--font", default=None)
    ap.add_argument("--height", type=int, default=1080)
    args = ap.parse_args()

    if not args.video.is_file():
        raise SystemExit(f"missing rendered video: {args.video}")
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise SystemExit("ffmpeg not on PATH")

    spans = json.loads(args.labels.read_text(encoding="utf-8"))["spans"]
    graph = build_filter(spans, find_font(args.font), args.height)

    # The graph is far past any sane command-line length, so hand it to ffmpeg as a file.
    script = args.out.with_suffix(".filter.txt")
    script.write_text(graph, encoding="utf-8")
    cmd = [
        ffmpeg, "-y", "-i", str(args.video),
        "-filter_complex_script", str(script),
        "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p",
        str(args.out),
    ]
    print(" ".join(cmd), flush=True)
    result = subprocess.run(cmd, stderr=subprocess.STDOUT, stdout=subprocess.PIPE, text=True)
    if result.returncode != 0:
        sys.stdout.write(result.stdout[-4000:])
        raise SystemExit(f"ffmpeg failed ({result.returncode})")
    script.unlink(missing_ok=True)
    print(f"LABELLED {args.out}")


if __name__ == "__main__":
    main()
