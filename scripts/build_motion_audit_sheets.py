"""Build readable paired playback strips from a rendered blind motion audit."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from urllib.parse import urlencode

import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFont


_SKELETON_EDGES = (
    (0, 1), (0, 2), (0, 3),
    (1, 4), (4, 7), (7, 10),
    (2, 5), (5, 8), (8, 11),
    (3, 6), (6, 9), (9, 12), (12, 15),
    (9, 13), (13, 16), (16, 18), (18, 20),
    (9, 14), (14, 17), (17, 19), (19, 21),
)
_TRAIL_JOINTS = {
    0: "#00d7ff",
    7: "#64ff8f",
    8: "#64ff8f",
    20: "#ff65d8",
    21: "#ffc857",
}


def _direction_legend(view: str) -> str:
    if view == "front":
        return "<- DANCER RIGHT | DANCER LEFT ->"
    return "<- BACKWARD | FORWARD ->"


def _load_projected(
    audit_dir: Path,
    identifier: str,
    view: str,
    frames: int,
    *,
    required: bool,
) -> np.ndarray | None:
    path = audit_dir / f"{identifier}_{view}_ybot.npz"
    if not path.is_file():
        if required:
            raise RuntimeError(f"{path.name}: protocol-10 skeleton evidence is missing")
        return None
    with np.load(path, allow_pickle=False) as payload:
        if "projected" not in payload:
            raise RuntimeError(f"{path.name}: projected Y-Bot joints are missing")
        projected = np.asarray(payload["projected"], dtype=np.float32)
    if projected.shape != (frames, 22, 3):
        raise RuntimeError(f"{path.name}: invalid projected-joint shape {projected.shape}")
    if not np.isfinite(projected).all():
        raise RuntimeError(f"{path.name}: projected Y-Bot joints are non-finite")
    return projected


def _projected_point(
    projected: np.ndarray,
    frame: int,
    joint: int,
    size: tuple[int, int],
) -> tuple[int, int] | None:
    x, y = map(float, projected[frame, joint, :2])
    if not (-0.25 <= x <= 1.25 and -0.25 <= y <= 1.25):
        return None
    width, height = size
    return (
        int(round(np.clip(x, 0.0, 1.0) * (width - 1))),
        int(round((1.0 - np.clip(y, 0.0, 1.0)) * (height - 1))),
    )


def _draw_direction_legend(image: Image.Image, view: str) -> None:
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    text = _direction_legend(view)
    box = draw.textbbox((0, 0), text, font=font)
    width = box[2] - box[0] + 10
    height = box[3] - box[1] + 6
    x = max(4, image.width - width - 4)
    y = max(4, image.height - height - 4)
    draw.rectangle((x, y, x + width, y + height), fill="#101018", outline="#f3d35b")
    draw.text((x + 5, y + 3), text, fill="#f3d35b", font=font)


def _draw_motion_evidence(
    image: Image.Image,
    projected: np.ndarray,
    frame: int,
    *,
    start: int,
    end: int,
) -> None:
    draw = ImageDraw.Draw(image)
    size = image.size
    start_frame = int(np.clip(start, 0, len(projected) - 1))
    end_frame = int(np.clip(max(start + 1, end), 1, len(projected)))

    feet = [
        _projected_point(projected, start_frame, joint, size)
        for joint in (7, 8, 10, 11)
    ]
    feet = [point for point in feet if point is not None]
    if feet:
        floor_y = max(point[1] for point in feet)
        draw.line((0, floor_y, image.width - 1, floor_y), fill="#5ad9ff", width=2)

    root_start = _projected_point(projected, start_frame, 0, size)
    if root_start is not None:
        for y in range(0, image.height, 12):
            draw.line(
                (root_start[0], y, root_start[0], min(image.height - 1, y + 6)),
                fill="#5ad9ff",
                width=1,
            )

    for joint, color in _TRAIL_JOINTS.items():
        points = [
            _projected_point(projected, trail_frame, joint, size)
            for trail_frame in range(start_frame, end_frame)
        ]
        points = [point for point in points if point is not None]
        if len(points) >= 2:
            draw.line(points, fill=color, width=2)
        if points:
            x, y = points[-1]
            draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill=color, outline="black")

    current = {
        joint: _projected_point(projected, frame, joint, size)
        for joint in range(22)
    }
    for parent, child in _SKELETON_EDGES:
        if current[parent] is not None and current[child] is not None:
            draw.line((current[parent], current[child]), fill="white", width=2)
    for joint, point in current.items():
        if point is None:
            continue
        x, y = point
        color = _TRAIL_JOINTS.get(joint, "#ffffff")
        radius = 3 if joint in _TRAIL_JOINTS else 2
        draw.ellipse(
            (x - radius, y - radius, x + radius, y + radius),
            fill=color,
            outline="black",
        )


def _frame_image(
    path: Path,
    *,
    projected: np.ndarray | None,
    frame: int,
    start: int,
    end: int,
    view: str,
    detail: bool,
) -> Image.Image:
    image = Image.open(path).convert("RGB")
    if projected is not None:
        _draw_motion_evidence(image, projected, frame, start=start, end=end)
    if detail:
        image = _detail_crop(image)
    _draw_direction_legend(image, view)
    return image


def _detail_crop(image: Image.Image) -> Image.Image:
    width, height = image.size
    return image.crop((
        int(round(0.18 * width)),
        int(round(0.02 * height)),
        int(round(0.82 * width)),
        int(round(0.74 * height)),
    ))


def _selected_frames(item: dict, *, stride: int, context: int) -> list[int]:
    frame_count = int(item["frames"])
    start, end = map(int, item["action_range"])
    event = int(item["event_frame"])
    lo = max(0, start - max(0, int(context)))
    hi = min(frame_count, end + max(0, int(context)))
    selected = set(range(lo, hi, max(1, int(stride))))
    selected.update((lo, start, event, max(start, end - 1), max(lo, hi - 1)))
    return sorted(frame for frame in selected if 0 <= frame < frame_count)


def _tile(
    canvas: Image.Image,
    draw: ImageDraw.ImageDraw,
    path: Path,
    *,
    x: int,
    y: int,
    size: int,
    label: str,
    active: bool,
    event: bool,
    font: ImageFont.ImageFont,
    detail: bool,
    projected: np.ndarray | None,
    frame: int,
    action_range: tuple[int, int],
    view: str,
) -> None:
    image = _frame_image(
        path,
        projected=projected,
        frame=frame,
        start=action_range[0],
        end=action_range[1],
        view=view,
        detail=detail,
    )
    image.thumbnail((size, size), Image.Resampling.LANCZOS)
    canvas.paste(image, (x, y))
    color, border = "#666", 1
    if active:
        color, border = "#a98cff", 3
    if event:
        color, border = "#ff6868", 5
    draw.rectangle(
        (x, y, x + size - 1, y + size - 1),
        outline=color,
        width=border,
    )
    draw.text(
        (x + 6, y + 6),
        label,
        fill="white",
        stroke_width=2,
        stroke_fill="black",
        font=font,
    )


def _build_view_sheet(
    audit_dir: Path,
    item: dict,
    *,
    view: str,
    selected: list[int],
    output: Path,
    columns: int,
    size: int,
    projected: dict[tuple[str, str], np.ndarray | None],
    detail: bool = False,
) -> None:
    take = item["take"]
    control = item["control"]
    start, end = map(int, item["action_range"])
    event = int(item["event_frame"])
    chunks = [selected[index:index + columns] for index in range(0, len(selected), columns)]
    header = 42
    row_header = 24
    row_height = row_header + size
    canvas = Image.new(
        "RGB",
        (columns * size, header + len(chunks) * 2 * row_height),
        "#111",
    )
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    draw.text(
        (10, 14),
        f"{take} | {view}{' detail' if detail else ''} | paired playback strip | answer hidden",
        fill="white",
        font=font,
    )
    draw.text((10, 28), _direction_legend(view), fill="#f3d35b", font=font)
    roots = (
        ("EDIT", take, audit_dir / f"{take}_{view}_frames"),
        ("SOURCE", control, audit_dir / f"{control}_{view}_frames"),
    )
    for chunk_index, chunk in enumerate(chunks):
        for pair_index, (kind, pair_id, root) in enumerate(roots):
            row = chunk_index * 2 + pair_index
            y = header + row * row_height
            draw.text((6, y + 6), kind, fill="#ddd", font=font)
            y += row_header
            for column, frame in enumerate(chunk):
                _tile(
                    canvas,
                    draw,
                    root / f"frame_{frame:05d}.png",
                    x=column * size,
                    y=y,
                    size=size,
                    label=f"{kind[0]}{frame}",
                    active=start <= frame < end,
                    event=frame == event,
                    font=font,
                    detail=detail,
                    projected=projected[(pair_id, view)],
                    frame=frame,
                    action_range=(start, end),
                    view=view,
                )
    suffix = f"{view}_detail" if detail else view
    canvas.save(output / f"{take}_{suffix}.jpg", quality=95)


def _dual_tile(
    canvas: Image.Image,
    draw: ImageDraw.ImageDraw,
    roots: tuple[Path, Path],
    *,
    x: int,
    y: int,
    size: int,
    label: str,
    active: bool,
    event: bool,
    font: ImageFont.ImageFont,
    projected: tuple[np.ndarray | None, np.ndarray | None],
    frame: int,
    action_range: tuple[int, int],
) -> None:
    for view_index, (view, root, evidence) in enumerate(
        zip(("front", "side"), roots, projected, strict=True)
    ):
        image = _frame_image(
            root,
            projected=evidence,
            frame=frame,
            start=action_range[0],
            end=action_range[1],
            view=view,
            detail=False,
        )
        image.thumbnail((size, size), Image.Resampling.LANCZOS)
        tile_x = x + view_index * size
        canvas.paste(image, (tile_x, y))
        draw.text(
            (tile_x + 6, y + 6),
            f"{label} {view[0].upper()}",
            fill="white",
            stroke_width=2,
            stroke_fill="black",
            font=font,
        )
    color, border = "#666", 1
    if active:
        color, border = "#a98cff", 3
    if event:
        color, border = "#ff6868", 5
    draw.rectangle(
        (x, y, x + 2 * size - 1, y + size - 1),
        outline=color,
        width=border,
    )
    draw.line((x + size, y, x + size, y + size - 1), fill="#888", width=1)


def _build_dual_sheet(
    audit_dir: Path,
    item: dict,
    *,
    selected: list[int],
    output: Path,
    columns: int,
    size: int,
    projected: dict[tuple[str, str], np.ndarray | None],
) -> None:
    take = item["take"]
    control = item["control"]
    start, end = map(int, item["action_range"])
    event = int(item["event_frame"])
    dual_columns = max(1, columns // 2)
    chunks = [
        selected[index:index + dual_columns]
        for index in range(0, len(selected), dual_columns)
    ]
    header = 42
    row_header = 24
    row_height = row_header + size
    canvas = Image.new(
        "RGB",
        (dual_columns * 2 * size, header + len(chunks) * 2 * row_height),
        "#111",
    )
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    draw.text(
        (10, 14),
        f"{take} | front + side | paired playback strip | answer hidden",
        fill="white",
        font=font,
    )
    draw.text(
        (10, 28),
        f"FRONT {_direction_legend('front')} | SIDE {_direction_legend('side')}",
        fill="#f3d35b",
        font=font,
    )
    pair_ids = (("EDIT", take), ("SOURCE", control))
    for chunk_index, chunk in enumerate(chunks):
        for pair_index, (kind, pair_id) in enumerate(pair_ids):
            row = chunk_index * 2 + pair_index
            y = header + row * row_height
            draw.text((6, y + 6), kind, fill="#ddd", font=font)
            y += row_header
            for column, frame in enumerate(chunk):
                _dual_tile(
                    canvas,
                    draw,
                    (
                        audit_dir / f"{pair_id}_front_frames" / f"frame_{frame:05d}.png",
                        audit_dir / f"{pair_id}_side_frames" / f"frame_{frame:05d}.png",
                    ),
                    x=column * 2 * size,
                    y=y,
                    size=size,
                    label=f"{kind[0]}{frame}",
                    active=start <= frame < end,
                    event=frame == event,
                    font=font,
                    projected=(
                        projected[(pair_id, "front")],
                        projected[(pair_id, "side")],
                    ),
                    frame=frame,
                    action_range=(start, end),
                )
    canvas.save(output / f"{take}_dual.jpg", quality=95)


def _review_tile(
    canvas: Image.Image,
    draw: ImageDraw.ImageDraw,
    roots: tuple[Path, Path, Path],
    *,
    x: int,
    y: int,
    size: int,
    label: str,
    active: bool,
    event: bool,
    font: ImageFont.ImageFont,
    projected: tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None],
    frame: int,
    action_range: tuple[int, int],
) -> None:
    for view_index, (view, root, evidence) in enumerate(
        zip(("front", "side", "front"), roots, projected, strict=True)
    ):
        detail = view_index == 2
        image = _frame_image(
            root,
            projected=evidence,
            frame=frame,
            start=action_range[0],
            end=action_range[1],
            view=view,
            detail=detail,
        )
        image.thumbnail((size, size), Image.Resampling.LANCZOS)
        tile_x = x + view_index * size
        canvas.paste(image, (tile_x, y))
        draw.text(
            (tile_x + 6, y + 6),
            f"{label} {('D' if detail else view[0].upper())}",
            fill="white",
            stroke_width=2,
            stroke_fill="black",
            font=font,
        )
    color, border = "#666", 1
    if active:
        color, border = "#a98cff", 3
    if event:
        color, border = "#ff6868", 5
    draw.rectangle(
        (x, y, x + 3 * size - 1, y + size - 1),
        outline=color,
        width=border,
    )
    for divider in (x + size, x + 2 * size):
        draw.line((divider, y, divider, y + size - 1), fill="#888", width=1)


def _delta_image(
    edit_path: Path,
    source_path: Path,
    *,
    detail: bool,
    view: str,
) -> Image.Image:
    edit = Image.open(edit_path).convert("RGB")
    source = Image.open(source_path).convert("RGB")
    if detail:
        edit = _detail_crop(edit)
        source = _detail_crop(source)
    difference = ImageChops.difference(edit, source)
    difference = Image.eval(difference, lambda value: min(255, value * 4))
    _draw_direction_legend(difference, view)
    return difference


def _delta_tile(
    canvas: Image.Image,
    draw: ImageDraw.ImageDraw,
    roots: tuple[tuple[Path, Path], tuple[Path, Path], tuple[Path, Path]],
    *,
    x: int,
    y: int,
    size: int,
    label: str,
    active: bool,
    event: bool,
    font: ImageFont.ImageFont,
) -> None:
    for view_index, (view, (edit_path, source_path)) in enumerate(
        zip(("F", "S", "D"), roots, strict=True)
    ):
        image = _delta_image(
            edit_path,
            source_path,
            detail=view == "D",
            view="side" if view == "S" else "front",
        )
        image.thumbnail((size, size), Image.Resampling.LANCZOS)
        tile_x = x + view_index * size
        canvas.paste(image, (tile_x, y))
        draw.text(
            (tile_x + 6, y + 6),
            f"{label} Δ{view}",
            fill="white",
            stroke_width=2,
            stroke_fill="black",
            font=font,
        )
    color, border = "#666", 1
    if active:
        color, border = "#a98cff", 3
    if event:
        color, border = "#ff6868", 5
    draw.rectangle(
        (x, y, x + 3 * size - 1, y + size - 1),
        outline=color,
        width=border,
    )
    for divider in (x + size, x + 2 * size):
        draw.line((divider, y, divider, y + size - 1), fill="#888", width=1)


def _build_review_pages(
    audit_dir: Path,
    item: dict,
    *,
    selected: list[int],
    output: Path,
    size: int,
    audit_id: str,
    motion_fingerprint: str,
    projected: dict[tuple[str, str], np.ndarray | None],
    page_frames: int = 4,
) -> None:
    take = item["take"]
    control = item["control"]
    start, end = map(int, item["action_range"])
    event = int(item["event_frame"])
    chunks = [
        selected[index:index + max(1, int(page_frames))]
        for index in range(0, len(selected), max(1, int(page_frames)))
    ]
    header = 42
    row_header = 24
    row_height = row_header + size
    pair_ids = (("EDIT", take), ("SOURCE", control))
    page_names = []
    for page_index, chunk in enumerate(chunks, start=1):
        canvas = Image.new(
            "RGB",
            (len(chunk) * 3 * size, header + 3 * row_height),
            "#111",
        )
        draw = ImageDraw.Draw(canvas)
        font = ImageFont.load_default()
        draw.text(
            (10, 14),
            (
                f"{take} | front + side + detail | page {page_index}/{len(chunks)}"
                " | answer hidden"
            ),
            fill="white",
            font=font,
        )
        draw.text(
            (10, 28),
            f"FRONT {_direction_legend('front')} | SIDE {_direction_legend('side')}",
            fill="#f3d35b",
            font=font,
        )
        for pair_index, (kind, pair_id) in enumerate(pair_ids):
            y = header + pair_index * row_height
            draw.text((6, y + 6), kind, fill="#ddd", font=font)
            y += row_header
            for column, frame in enumerate(chunk):
                front = (
                    audit_dir
                    / f"{pair_id}_front_frames"
                    / f"frame_{frame:05d}.png"
                )
                _review_tile(
                    canvas,
                    draw,
                    (
                        front,
                        audit_dir
                        / f"{pair_id}_side_frames"
                        / f"frame_{frame:05d}.png",
                        front,
                    ),
                    x=column * 3 * size,
                    y=y,
                    size=size,
                    label=f"{kind[0]}{frame}",
                    active=start <= frame < end,
                    event=frame == event,
                    font=font,
                    projected=(
                        projected[(pair_id, "front")],
                        projected[(pair_id, "side")],
                        projected[(pair_id, "front")],
                    ),
                    frame=frame,
                    action_range=(start, end),
                )
        delta_y = header + 2 * row_height
        draw.text((6, delta_y + 6), "EDIT - SOURCE", fill="#ddd", font=font)
        delta_y += row_header
        for column, frame in enumerate(chunk):
            edit_front = (
                audit_dir
                / f"{take}_front_frames"
                / f"frame_{frame:05d}.png"
            )
            source_front = (
                audit_dir
                / f"{control}_front_frames"
                / f"frame_{frame:05d}.png"
            )
            edit_side = (
                audit_dir
                / f"{take}_side_frames"
                / f"frame_{frame:05d}.png"
            )
            source_side = (
                audit_dir
                / f"{control}_side_frames"
                / f"frame_{frame:05d}.png"
            )
            _delta_tile(
                canvas,
                draw,
                (
                    (edit_front, source_front),
                    (edit_side, source_side),
                    (edit_front, source_front),
                ),
                x=column * 3 * size,
                y=delta_y,
                size=size,
                label=f"Δ{frame}",
                active=start <= frame < end,
                event=frame == event,
                font=font,
            )
        page_name = f"{take}_review_{page_index:02d}.jpg"
        canvas.save(output / page_name, quality=95)
        page_names.append(page_name)
    cache_query = html.escape(
        urlencode({
            "audit_id": str(audit_id),
            "motion_fingerprint": str(motion_fingerprint),
            "take_id": str(take),
        }),
        quote=True,
    )
    page_images = "\n".join(
        f'<img src="{page_name}?{cache_query}" alt="{take} review page {index}">'
        for index, page_name in enumerate(page_names, start=1)
    )
    acknowledgment = json.dumps({
        "type": "maestro-motion-audit-comparison-ready",
        "auditId": str(audit_id),
        "motionFingerprint": str(motion_fingerprint),
        "takeId": str(take),
    })
    (output / f"{take}_review.html").write_text(
        (
            "<!doctype html><meta charset=\"utf-8\">"
            f"<title>{take} synchronized review</title>"
            "<style>body{margin:0;background:#111;color:#eee;font-family:system-ui,sans-serif}"
            "h1,p{padding:12px 18px;margin:0}.legend{position:sticky;top:0;z-index:2;"
            "padding:10px 18px;background:#241f12;color:#f3d35b;font-weight:700}"
            "img{display:block;width:100%;height:auto;margin:0 0 12px}"
            "</style>"
            f"<h1>{take}: synchronized front, side, upper-body, "
            "and edit-minus-source review</h1>"
            "<div class=\"legend\">FRONT: screen-left = dancer right; "
            "screen-right = dancer left. SIDE: screen-left = backward; "
            "screen-right = forward. White skeleton = exact rendered Y-Bot joints; "
            "cyan/magenta/yellow/green trails = root, hands, and feet.</div>"
            "<p id=\"comparison-status\">Loading comparison evidence...</p>"
            f"{page_images}"
            "<script>"
            f"const acknowledgment={acknowledgment};"
            "window.addEventListener('load',()=>{"
            "const status=document.querySelector('#comparison-status');"
            "const query=new URLSearchParams(window.location.search);"
            "const queryMatches="
            "query.get('audit_id')===acknowledgment.auditId&&"
            "query.get('motion_fingerprint')===acknowledgment.motionFingerprint&&"
            "query.get('take_id')===acknowledgment.takeId;"
            "const imagesLoaded=[...document.images].every(image=>"
            "image.complete&&image.naturalWidth>0);"
            "if(!queryMatches){status.textContent='Comparison identity mismatch; review not recorded.';return;}"
            "if(!imagesLoaded){status.textContent='Comparison evidence failed to load; review not recorded.';return;}"
            "if(!window.opener){status.textContent='Parent audit page is unavailable; review not recorded.';return;}"
            "const targetOrigin=window.location.origin==='null'?'*':window.location.origin;"
            "window.opener.postMessage(acknowledgment,targetOrigin);"
            "status.textContent='Comparison evidence loaded; acknowledgment sent to the audit page.';"
            "});"
            "</script>"
        ),
        encoding="utf-8",
    )


def build_sheets(
    audit_dir: Path,
    *,
    stride: int = 1,
    context: int = 2,
    columns: int = 10,
    size: int = 180,
) -> Path:
    review = json.loads((audit_dir / "review.json").read_text(encoding="utf-8"))
    output = audit_dir / "phase_sheets"
    output.mkdir(exist_ok=True)
    protocol_nine = int(review.get("review_protocol_version", 0)) >= 9
    frame_counts = {
        item["take"]: int(item["frames"])
        for item in review["takes"]
    }
    frame_counts.update({
        item["control"]: int(item["frames"])
        for item in review.get("controls", ())
    })
    for item in review["takes"]:
        frame_counts.setdefault(item["control"], int(item["frames"]))
    projected = {
        (identifier, view): _load_projected(
            audit_dir,
            identifier,
            view,
            frames,
            required=protocol_nine,
        )
        for identifier, frames in frame_counts.items()
        for view in ("front", "side")
    }
    for item in review["takes"]:
        selected = _selected_frames(item, stride=stride, context=context)
        for view in ("front", "side"):
            _build_view_sheet(
                audit_dir,
                item,
                view=view,
                selected=selected,
                output=output,
                columns=max(1, int(columns)),
                size=max(96, int(size)),
                projected=projected,
            )
        _build_view_sheet(
            audit_dir,
            item,
            view="front",
            selected=selected,
            output=output,
            columns=max(1, int(columns)),
            size=max(96, int(size)),
            projected=projected,
            detail=True,
        )
        _build_dual_sheet(
            audit_dir,
            item,
            selected=selected,
            output=output,
            columns=max(1, int(columns)),
            size=max(96, int(size)),
            projected=projected,
        )
        _build_review_pages(
            audit_dir,
            item,
            selected=selected,
            output=output,
            size=max(96, int(size)),
            audit_id=str(review["audit_id"]),
            motion_fingerprint=str(review["motion_fingerprint"]),
            projected=projected,
        )
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("audit_dir", type=Path)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--context", type=int, default=2)
    parser.add_argument("--columns", type=int, default=10)
    parser.add_argument("--size", type=int, default=180)
    args = parser.parse_args()
    output = build_sheets(
        args.audit_dir,
        stride=args.stride,
        context=args.context,
        columns=args.columns,
        size=args.size,
    )
    print(f"built paired phase sheets in {output}")


if __name__ == "__main__":
    main()
