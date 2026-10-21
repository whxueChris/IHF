from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter

import plot_emoji_localization_publication_ready as base
import plot_low_vision_high_myopia_localization_publication_ready as source


OUTPUT_DIR = Path(__file__).resolve().parent / "output"

PANEL_WIDTH_MM = 44.0
PANEL_HEIGHT_MM = 40.0
COMBINED_WIDTH_MM = 183.0
COMBINED_HEIGHT_MM = 102.0

BLUE = (33, 102, 172)
RED = (217, 95, 95)
INK = (34, 34, 34)
WHITE = (255, 255, 255)

TITLE_FONT = base.font_pt(9.0, True)
CONDITION_FONT = base.font_pt(7.5, False)
LEGEND_FONT = base.font_pt(7.5, False)
FEATURE_FONT = base.font_pt(7.0, False)
SHARED_TITLE_FONT = base.font_pt(9.0, True)
COLUMN_FONT = base.font_pt(8.5, True)
ROW_FONT = base.font_pt(8.0, True)
COMBINED_FEATURE_FONT = base.font_pt(6.5, False)
COMBINED_LEGEND_FONT = base.font_pt(8.5, False)


def mm(value: float) -> float:
    return base.mm_to_px(value)


def text_center(
    draw: ImageDraw.ImageDraw,
    x: float,
    y: float,
    value: str,
    font,
    fill: tuple[int, int, int] = INK,
) -> None:
    draw.text((x, y), value, font=font, fill=fill, anchor="mm")


def draw_star(draw: ImageDraw.ImageDraw, x: float, y: float, radius: float) -> None:
    points = base.star_points(x, y, radius)
    draw.polygon(points, fill=WHITE)
    draw.line(
        points + [points[0]],
        fill=INK,
        width=max(1, round(base.pt_to_px(0.45))),
        joint="curve",
    )


def draw_text_halo(draw: ImageDraw.ImageDraw, x: float, y: float, value: str) -> None:
    halo = max(1, round(base.pt_to_px(0.3)))
    for dx, dy in [(-halo, 0), (halo, 0), (0, -halo), (0, halo)]:
        draw.text(
            (x + dx, y + dy),
            value,
            font=FEATURE_FONT,
            fill=WHITE,
            anchor="mm",
        )
    draw.text((x, y), value, font=FEATURE_FONT, fill=INK, anchor="mm")


def draw_legend(draw: ImageDraw.ImageDraw, y: float) -> None:
    items = [
        (mm(4.0), "Density", RED),
        (mm(22.0), "Response", BLUE),
        (mm(43.0), "Target", INK),
    ]
    for x, label, color in items:
        if label == "Target":
            draw_star(draw, x, y, mm(1.0))
        else:
            radius = mm(0.75 if label == "Response" else 0.9)
            draw.ellipse(
                (x - radius, y - radius, x + radius, y + radius),
                fill=color,
                outline=WHITE,
            )
        draw.text(
            (x + mm(2.0), y),
            label,
            font=LEGEND_FONT,
            fill=INK,
            anchor="lm",
        )


def draw_heatmap(
    stimulus: str,
    condition: str,
    rows: list[dict[str, object]],
    gt: dict[str, dict[str, tuple[float, float]]],
    emoji_images: dict[str, Image.Image],
    source_bounds: tuple[float, float, float, float],
) -> Image.Image:
    width = round(mm(PANEL_WIDTH_MM))
    height = round(mm(PANEL_HEIGHT_MM))
    canvas = Image.new("RGBA", (width, height), (*WHITE, 0))
    draw = ImageDraw.Draw(canvas)

    panel_rows = [
        row
        for row in rows
        if row["stimulus"] == stimulus and row["condition"] == condition
    ]
    heat_box = (mm(1.0), mm(1.0), width - mm(1.0), height - mm(1.0))
    transform = base.transform_factory(source_bounds, heat_box)
    base.paste_emoji(
        canvas,
        stimulus,
        transform,
        gt,
        emoji_images,
        alpha_scale=0.30,
    )

    density = Image.new("RGBA", canvas.size, (255, 255, 255, 0))
    density_draw = ImageDraw.Draw(density)
    density_radius = mm(3.8)
    for row in panel_rows:
        x, y = transform(
            float(row["marker_window_x_px"]),
            float(row["marker_window_y_px"]),
        )
        density_draw.ellipse(
            (x - density_radius, y - density_radius, x + density_radius, y + density_radius),
            fill=(*RED, 68),
        )
    density = density.filter(
        ImageFilter.GaussianBlur(max(1, round(mm(1.75))))
    )
    canvas.alpha_composite(density)

    response_radius = mm(0.50)
    response_outline = max(1, round(base.pt_to_px(0.30)))
    for row in panel_rows:
        x, y = transform(
            float(row["marker_window_x_px"]),
            float(row["marker_window_y_px"]),
        )
        draw.ellipse(
            (
                x - response_radius,
                y - response_radius,
                x + response_radius,
                y + response_radius,
            ),
            fill=BLUE,
            outline=WHITE,
            width=response_outline,
        )

    for feature in base.FEATURE_ORDER:
        tx, ty = gt[stimulus][feature]
        x, y = transform(tx, ty)
        draw_star(draw, x, y, mm(1.3))

    return canvas


def draw_combined_text_halo(
    draw: ImageDraw.ImageDraw,
    x: float,
    y: float,
    value: str,
) -> None:
    halo = max(1, round(base.pt_to_px(0.25)))
    for dx, dy in [(-halo, 0), (halo, 0), (0, -halo), (0, halo)]:
        draw.text(
            (x + dx, y + dy),
            value,
            font=COMBINED_FEATURE_FONT,
            fill=WHITE,
            anchor="mm",
        )
    draw.text(
        (x, y),
        value,
        font=COMBINED_FEATURE_FONT,
        fill=INK,
        anchor="mm",
    )


def draw_combined_legend(
    draw: ImageDraw.ImageDraw,
    width: float,
    y: float,
) -> None:
    items = [
        (width / 2 - mm(30.0), "Density", RED),
        (width / 2, "Response", BLUE),
        (width / 2 + mm(31.0), "Target", INK),
    ]
    for x, label, color in items:
        if label == "Target":
            draw_star(draw, x, y, mm(0.9))
        else:
            radius = mm(0.68 if label == "Response" else 0.82)
            draw.ellipse(
                (x - radius, y - radius, x + radius, y + radius),
                fill=color,
                outline=WHITE,
            )
        draw.text(
            (x + mm(1.8), y),
            label,
            font=COMBINED_LEGEND_FONT,
            fill=INK,
            anchor="lm",
        )


def draw_combined_panel(
    canvas: Image.Image,
    stimulus: str,
    condition: str,
    box: tuple[float, float, float, float],
    rows: list[dict[str, object]],
    gt: dict[str, dict[str, tuple[float, float]]],
    emoji_images: dict[str, Image.Image],
    source_bounds: tuple[float, float, float, float],
) -> None:
    draw = ImageDraw.Draw(canvas)
    panel_rows = [
        row
        for row in rows
        if row["stimulus"] == stimulus and row["condition"] == condition
    ]
    transform = base.transform_factory(source_bounds, box)
    base.paste_emoji(
        canvas,
        stimulus,
        transform,
        gt,
        emoji_images,
        alpha_scale=0.30,
    )

    density = Image.new("RGBA", canvas.size, (255, 255, 255, 0))
    density_draw = ImageDraw.Draw(density)
    density_radius = mm(2.8)
    for row in panel_rows:
        x, y = transform(
            float(row["marker_window_x_px"]),
            float(row["marker_window_y_px"]),
        )
        density_draw.ellipse(
            (x - density_radius, y - density_radius, x + density_radius, y + density_radius),
            fill=(*RED, 62),
        )
    density = density.filter(
        ImageFilter.GaussianBlur(max(1, round(mm(1.35))))
    )
    canvas.alpha_composite(density)

    response_radius = mm(0.42)
    response_outline = max(1, round(base.pt_to_px(0.28)))
    for row in panel_rows:
        x, y = transform(
            float(row["marker_window_x_px"]),
            float(row["marker_window_y_px"]),
        )
        draw.ellipse(
            (
                x - response_radius,
                y - response_radius,
                x + response_radius,
                y + response_radius,
            ),
            fill=BLUE,
            outline=WHITE,
            width=response_outline,
        )

    for feature in base.FEATURE_ORDER:
        tx, ty = gt[stimulus][feature]
        x, y = transform(tx, ty)
        draw_star(draw, x, y, mm(1.05))


def draw_combined_heatmaps(
    rows: list[dict[str, object]],
    gt: dict[str, dict[str, tuple[float, float]]],
    emoji_images: dict[str, Image.Image],
    source_bounds: tuple[float, float, float, float],
) -> Image.Image:
    width = round(mm(COMBINED_WIDTH_MM))
    height = round(mm(COMBINED_HEIGHT_MM))
    canvas = Image.new("RGBA", (width, height), (*WHITE, 0))
    draw = ImageDraw.Draw(canvas)

    text_center(
        draw,
        width / 2,
        mm(3.8),
        "Residual-vision participants (n = 13)",
        SHARED_TITLE_FONT,
    )

    left_margin = mm(29.0)
    right_margin = mm(4.0)
    col_gap = mm(3.0)
    panel_width = (width - left_margin - right_margin - 2 * col_gap) / 3
    row_boxes = [
        (mm(14.0), mm(51.5)),
        (mm(54.5), mm(92.0)),
    ]

    for col_idx, stimulus in enumerate(base.EMOJI_ORDER):
        left = left_margin + col_idx * (panel_width + col_gap)
        center_x = left + panel_width / 2
        text_center(
            draw,
            center_x,
            mm(10.0),
            stimulus.title(),
            COLUMN_FONT,
        )

    for row_idx, condition in enumerate(source.CONDITIONS):
        top, bottom = row_boxes[row_idx]
        text_center(
            draw,
            mm(15.0),
            (top + bottom) / 2,
            source.CONDITION_LABELS[condition],
            ROW_FONT,
        )
        for col_idx, stimulus in enumerate(base.EMOJI_ORDER):
            left = left_margin + col_idx * (panel_width + col_gap)
            box = (left, top, left + panel_width, bottom)
            draw_combined_panel(
                canvas,
                stimulus,
                condition,
                box,
                rows,
                gt,
                emoji_images,
                source_bounds,
            )

    draw_combined_legend(draw, width, mm(97.5))
    return canvas


def prepare_output_directory() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    resolved_output = OUTPUT_DIR.resolve()
    if resolved_output != (Path(__file__).resolve().parent / "output").resolve():
        raise RuntimeError(
            f"Refusing to clean unexpected output directory: {resolved_output}"
        )
    for path in resolved_output.iterdir():
        if path.is_file():
            path.unlink()


def save_png(canvas: Image.Image, filename: str) -> Path:
    path = OUTPUT_DIR / filename
    canvas.save(
        path,
        format="PNG",
        dpi=(base.EXPORT_DPI, base.EXPORT_DPI),
    )
    return path


def validate(rows: list[dict[str, object]]) -> None:
    participants = {str(row["participant_id"]) for row in rows}
    if len(participants) != 13:
        raise ValueError(f"Expected 13 residual-vision participants, found {len(participants)}")
    for condition in source.CONDITIONS:
        for stimulus in base.EMOJI_ORDER:
            for feature in base.FEATURE_ORDER:
                ids = {
                    str(row["participant_id"])
                    for row in rows
                    if row["condition"] == condition
                    and row["stimulus"] == stimulus
                    and row["feature"] == feature
                }
                if len(ids) != 13:
                    raise ValueError(
                        f"Expected n=13 for {condition}/{stimulus}/{feature}, found {len(ids)}"
                    )


def main() -> None:
    prepare_output_directory()
    gt = base.read_gt()
    wide_rows = source.read_wide_rows()
    rows = source.wide_to_long(wide_rows)
    validate(rows)
    emoji_images = base.read_html_emoji_images()
    source_bounds = base.global_source_bounds(rows, gt)

    outputs: list[Path] = []
    for condition in source.CONDITIONS:
        for stimulus in base.EMOJI_ORDER:
            filename = f"{condition}_{stimulus}_heatmap.png"
            outputs.append(
                save_png(
                    draw_heatmap(
                        stimulus,
                        condition,
                        rows,
                        gt,
                        emoji_images,
                        source_bounds,
                    ),
                    filename,
                )
            )

    outputs.append(
        save_png(
            draw_combined_heatmaps(
                rows,
                gt,
                emoji_images,
                source_bounds,
            ),
            "residual_vision_heatmaps_combined.png",
        )
    )

    final_files = sorted(path for path in OUTPUT_DIR.iterdir() if path.is_file())
    if len(final_files) != 7 or any(
        path.suffix.lower() != ".png" for path in final_files
    ):
        raise RuntimeError("Final output directory must contain exactly seven PNG files")

    print(f"Output directory: {OUTPUT_DIR}")
    for path in outputs:
        print(path.name)


if __name__ == "__main__":
    main()
