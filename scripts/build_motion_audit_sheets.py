"""Build readable paired playback strips from a rendered blind motion audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageChops, ImageDraw, ImageFont


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
) -> None:
    image = Image.open(path).convert("RGB")
    if detail:
        image = _detail_crop(image)
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
    roots = (
        ("EDIT", audit_dir / f"{take}_{view}_frames"),
        ("SOURCE", audit_dir / f"{control}_{view}_frames"),
    )
    for chunk_index, chunk in enumerate(chunks):
        for pair_index, (kind, root) in enumerate(roots):
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
) -> None:
    for view_index, (view, root) in enumerate(zip(("F", "S"), roots, strict=True)):
        image = Image.open(root).convert("RGB")
        image.thumbnail((size, size), Image.Resampling.LANCZOS)
        tile_x = x + view_index * size
        canvas.paste(image, (tile_x, y))
        draw.text(
            (tile_x + 6, y + 6),
            f"{label} {view}",
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
) -> None:
    for view_index, (view, root) in enumerate(
        zip(("F", "S", "D"), roots, strict=True)
    ):
        image = Image.open(root).convert("RGB")
        if view == "D":
            image = _detail_crop(image)
        image.thumbnail((size, size), Image.Resampling.LANCZOS)
        tile_x = x + view_index * size
        canvas.paste(image, (tile_x, y))
        draw.text(
            (tile_x + 6, y + 6),
            f"{label} {view}",
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


def _delta_image(edit_path: Path, source_path: Path, *, detail: bool) -> Image.Image:
    edit = Image.open(edit_path).convert("RGB")
    source = Image.open(source_path).convert("RGB")
    if detail:
        edit = _detail_crop(edit)
        source = _detail_crop(source)
    difference = ImageChops.difference(edit, source)
    return Image.eval(difference, lambda value: min(255, value * 4))


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
    page_images = "\n".join(
        f'<img src="{page_name}" alt="{take} review page {index}">'
        for index, page_name in enumerate(page_names, start=1)
    )
    (output / f"{take}_review.html").write_text(
        (
            "<!doctype html><meta charset=\"utf-8\">"
            f"<title>{take} synchronized review</title>"
            "<style>body{margin:0;background:#111;color:#eee;font-family:system-ui,sans-serif}"
            "h1{padding:12px 18px;margin:0}img{display:block;width:100%;height:auto;margin:0 0 12px}"
            "</style>"
            f"<h1>{take}: synchronized front, side, upper-body, "
            "and edit-minus-source review</h1>"
            f"{page_images}"
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
            )
        _build_view_sheet(
            audit_dir,
            item,
            view="front",
            selected=selected,
            output=output,
            columns=max(1, int(columns)),
            size=max(96, int(size)),
            detail=True,
        )
        _build_dual_sheet(
            audit_dir,
            item,
            selected=selected,
            output=output,
            columns=max(1, int(columns)),
            size=max(96, int(size)),
        )
        _build_review_pages(
            audit_dir,
            item,
            selected=selected,
            output=output,
            size=max(96, int(size)),
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
