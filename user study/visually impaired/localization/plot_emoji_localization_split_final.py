from __future__ import annotations

import csv
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter

import plot_emoji_localization_publication_ready as base


OUTPUT_DIR = Path(__file__).resolve().parent / "output"
FIG6_DIR = Path(__file__).resolve().parents[2]
MANUAL_GROUND_TRUTH_CSV = Path(__file__).resolve().parent / 'emoji_localization_manual_ground_truth_manual_visual_annotation_2026-07-30T15-54-56_2026-07-30T15-57-28.csv'

# The Fig. 6a trial data are stored in the historical 665 × 440-pixel window.
# The final manual annotation expresses the same feature centres in the 410 ×
# 410-pixel stimulus coordinate system used for the Fig. 6c matched condition.
HISTORICAL_STIMULUS_LEFT_PX = 396.92
HISTORICAL_STIMULUS_TOP_PX = 182.99
HISTORICAL_STIMULUS_WIDTH_PX = 410.0
HISTORICAL_STIMULUS_HEIGHT_PX = 410.0

HEATMAP_WIDTH_MM = 44.0
HEATMAP_HEIGHT_MM = 40.0
ERROR_WIDTH_MM = 50.0
ERROR_HEIGHT_MM = 50.0

BLUE = (33, 102, 172)
RED = (217, 95, 95)
INK = (34, 34, 34)
AXIS = (70, 70, 70)
BOX_FILL = (198, 219, 239)
WHITE = (255, 255, 255)

TITLE_FONT = base.font_pt(7.0, True)
LEGEND_FONT = base.font_pt(5.5, False)
FEATURE_FONT = base.font_pt(6.0, False)
AXIS_FONT = base.font_pt(8.5, False)
TICK_FONT = base.font_pt(8.5, False)


def read_manual_registered_gt() -> dict[str, dict[str, tuple[float, float]]]:
    """Map the final manual feature centres into the Fig. 6a trial window."""
    gt: dict[str, dict[str, tuple[float, float]]] = {}
    with MANUAL_GROUND_TRUTH_CSV.open(newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            stimulus = row["stimulus_key"]
            feature = row["feature"]
            if stimulus not in base.EMOJI_ORDER or feature not in base.FEATURE_ORDER:
                continue
            gt.setdefault(stimulus, {})[feature] = (
                HISTORICAL_STIMULUS_LEFT_PX
                + float(row["true_target_x_norm"]) * HISTORICAL_STIMULUS_WIDTH_PX,
                HISTORICAL_STIMULUS_TOP_PX
                + float(row["true_target_y_norm"]) * HISTORICAL_STIMULUS_HEIGHT_PX,
            )

    missing = [
        f"{stimulus}-{feature}"
        for stimulus in base.EMOJI_ORDER
        for feature in base.FEATURE_ORDER
        if feature not in gt.get(stimulus, {})
    ]
    if missing:
        raise ValueError(f"Missing manual ground-truth coordinates: {', '.join(missing)}")
    return gt


def mm(value: float) -> float:
    return base.mm_to_px(value)


def text_center(draw: ImageDraw.ImageDraw, x: float, y: float, value: str, font, fill=INK) -> None:
    draw.text((x, y), value, font=font, fill=fill, anchor="mm")


def draw_star(draw: ImageDraw.ImageDraw, x: float, y: float, radius: float) -> None:
    points = base.star_points(x, y, radius)
    draw.polygon(points, fill=WHITE)
    draw.line(points + [points[0]], fill=INK, width=max(1, round(base.pt_to_px(0.45))), joint="curve")


def draw_text_halo(draw: ImageDraw.ImageDraw, x: float, y: float, value: str) -> None:
    halo = max(1, round(base.pt_to_px(0.3)))
    for dx, dy in [(-halo, 0), (halo, 0), (0, -halo), (0, halo)]:
        draw.text((x + dx, y + dy), value, font=FEATURE_FONT, fill=WHITE, anchor="mm")
    draw.text((x, y), value, font=FEATURE_FONT, fill=INK, anchor="mm")


def draw_heatmap_legend(draw: ImageDraw.ImageDraw, y: float) -> None:
    items = [
        (mm(4.0), "density", RED),
        (mm(22.0), "response", BLUE),
        (mm(43.0), "target", INK),
    ]
    for x, label, color in items:
        if label == "target":
            draw_star(draw, x, y, mm(1.0))
        else:
            radius = mm(0.75 if label == "response" else 0.9)
            draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color, outline=WHITE)
        draw.text((x + mm(2.0), y), label.capitalize(), font=LEGEND_FONT, fill=INK, anchor="lm")


def draw_heatmap(
    stimulus: str,
    rows: list[dict[str, object]],
    gt: dict[str, dict[str, tuple[float, float]]],
    emoji_images: dict[str, Image.Image],
    source_bounds: tuple[float, float, float, float],
) -> Image.Image:
    width = round(mm(HEATMAP_WIDTH_MM))
    height = round(mm(HEATMAP_HEIGHT_MM))
    canvas = Image.new("RGBA", (width, height), (*WHITE, 0))
    draw = ImageDraw.Draw(canvas)

    heat_box = (mm(1.0), mm(1.0), width - mm(1.0), height - mm(1.0))
    transform = base.transform_factory(source_bounds, heat_box)
    base.paste_emoji(canvas, stimulus, transform, gt, emoji_images, alpha_scale=0.30)

    density = Image.new("RGBA", canvas.size, (255, 255, 255, 0))
    density_draw = ImageDraw.Draw(density)
    density_radius = mm(4.0)
    for row in rows:
        if row["stimulus"] != stimulus:
            continue
        x, y = transform(float(row["marker_window_x_px"]), float(row["marker_window_y_px"]))
        density_draw.ellipse(
            (x - density_radius, y - density_radius, x + density_radius, y + density_radius),
            fill=(*RED, 96),
        )
    density = density.filter(ImageFilter.GaussianBlur(max(1, round(mm(1.8)))))
    canvas.alpha_composite(density)

    response_radius = mm(0.58)
    response_outline = max(1, round(base.pt_to_px(0.35)))
    for row in rows:
        if row["stimulus"] != stimulus:
            continue
        x, y = transform(float(row["marker_window_x_px"]), float(row["marker_window_y_px"]))
        draw.ellipse(
            (x - response_radius, y - response_radius, x + response_radius, y + response_radius),
            fill=BLUE,
            outline=WHITE,
            width=response_outline,
        )

    for feature in base.FEATURE_ORDER:
        tx, ty = gt[stimulus][feature]
        x, y = transform(tx, ty)
        draw_star(draw, x, y, mm(1.3))

    return canvas


def draw_rotated_axis_label(canvas: Image.Image, x: float, y: float) -> None:
    text = "Interaction error (%)"
    probe = ImageDraw.Draw(Image.new("RGBA", (1, 1)))
    bbox = probe.multiline_textbbox((0, 0), text, font=AXIS_FONT, spacing=2, align="center")
    layer_width = round(bbox[2] - bbox[0] + 24)
    layer_height = round(bbox[3] - bbox[1] + 24)
    layer = Image.new("RGBA", (layer_width, layer_height), (255, 255, 255, 0))
    layer_draw = ImageDraw.Draw(layer)
    layer_draw.multiline_text(
        (layer.width / 2, layer.height / 2),
        text,
        font=AXIS_FONT,
        fill=INK,
        anchor="mm",
        spacing=2,
        align="center",
    )
    layer = layer.rotate(90, expand=True)
    canvas.alpha_composite(layer, (round(x - layer.width / 2), round(y - layer.height / 2)))


def draw_error_plot(stimulus: str, rows: list[dict[str, object]], detail: list[dict[str, object]]) -> Image.Image:
    width = round(mm(ERROR_WIDTH_MM))
    height = round(mm(ERROR_HEIGHT_MM))
    canvas = Image.new("RGBA", (width, height), (*WHITE, 0))
    draw = ImageDraw.Draw(canvas)

    plot_left = mm(12.0)
    plot_right = width - mm(2.0)
    plot_top = mm(2.8)
    plot_bottom = mm(42.0)
    y_max = base.ERROR_Y_MAX_PERCENT
    axis_width = max(1, round(base.pt_to_px(0.55)))

    def y_pos(value: float) -> float:
        return plot_bottom - (value / y_max) * (plot_bottom - plot_top)

    draw.line((plot_left, plot_top, plot_left, plot_bottom), fill=AXIS, width=axis_width)
    draw.line((plot_left, plot_bottom, plot_right, plot_bottom), fill=AXIS, width=axis_width)

    for tick in base.ERROR_Y_TICKS_PERCENT:
        y = y_pos(float(tick))
        draw.line((plot_left - mm(0.9), y, plot_left, y), fill=AXIS, width=axis_width)
        draw.text((plot_left - mm(1.7), y), str(tick), font=TICK_FONT, fill=INK, anchor="rm")

    box_width = mm(6.2)
    cap_width = mm(3.6)
    dot_radius = mm(0.60)
    jitter = [-mm(1.15), mm(0.85), -mm(0.4), mm(1.3), -mm(1.55), mm(0.25), mm(1.65)]

    for index, feature in enumerate(base.FEATURE_ORDER):
        x = plot_left + (index + 0.5) * (plot_right - plot_left) / 3.0
        values = base.grouped_distances(detail, stimulus, feature)
        if not values:
            continue
        stats = base.box_stats(values)

        draw.line((x, y_pos(stats["min"]), x, y_pos(stats["max"])), fill=AXIS, width=axis_width)
        draw.line((x - cap_width / 2, y_pos(stats["min"]), x + cap_width / 2, y_pos(stats["min"])), fill=AXIS, width=axis_width)
        draw.line((x - cap_width / 2, y_pos(stats["max"]), x + cap_width / 2, y_pos(stats["max"])), fill=AXIS, width=axis_width)
        draw.rectangle(
            (x - box_width / 2, y_pos(stats["q3"]), x + box_width / 2, y_pos(stats["q1"])),
            fill=BOX_FILL,
            outline=INK,
            width=axis_width,
        )
        draw.line(
            (x - box_width / 2, y_pos(stats["median"]), x + box_width / 2, y_pos(stats["median"])),
            fill=INK,
            width=max(axis_width, round(base.pt_to_px(0.75))),
        )

        for point_index, value in enumerate(values):
            px = x + jitter[point_index % len(jitter)]
            py = y_pos(float(value))
            draw.ellipse(
                (px - dot_radius, py - dot_radius, px + dot_radius, py + dot_radius),
                fill=BLUE,
                outline=WHITE,
                width=max(1, round(base.pt_to_px(0.3))),
            )
        text_center(draw, x, mm(46.5), feature, TICK_FONT)

    draw_rotated_axis_label(canvas, mm(3.7), (plot_top + plot_bottom) / 2)
    return canvas


def prepare_output_directory() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    resolved_output = OUTPUT_DIR.resolve()
    if resolved_output != (Path(__file__).resolve().parent / "output").resolve():
        raise RuntimeError(f"Refusing to clean unexpected output directory: {resolved_output}")
    for path in resolved_output.iterdir():
        if path.is_file():
            path.unlink()


def save_png(canvas: Image.Image, filename: str) -> Path:
    path = OUTPUT_DIR / filename
    canvas.save(path, format="PNG", dpi=(base.EXPORT_DPI, base.EXPORT_DPI))
    return path


def main() -> None:
    prepare_output_directory()
    gt = read_manual_registered_gt()
    rows = base.read_marker_rows()
    detail = base.compute_distances(rows, gt)
    emoji_images = base.read_html_emoji_images()
    source_bounds = base.global_source_bounds(rows, gt)

    participants, counts = base.validate_counts(rows)
    if len(participants) != 7:
        raise ValueError(f"Expected 7 blind participants, found {len(participants)}")
    if any(counts[(stimulus, feature)] != 7 for stimulus in base.EMOJI_ORDER for feature in base.FEATURE_ORDER):
        raise ValueError("Expected n=7 for every emoji-feature combination")

    outputs: list[Path] = []
    for stimulus in base.EMOJI_ORDER:
        outputs.append(save_png(draw_heatmap(stimulus, rows, gt, emoji_images, source_bounds), f"{stimulus}_heatmap.png"))
        outputs.append(save_png(draw_error_plot(stimulus, rows, detail), f"{stimulus}_error.png"))

    final_files = sorted(path for path in OUTPUT_DIR.iterdir() if path.is_file())
    if len(final_files) != 6 or any(path.suffix.lower() != ".png" for path in final_files):
        raise RuntimeError("Final output directory must contain exactly six PNG files")

    print(f"Output directory: {OUTPUT_DIR}")
    for path in outputs:
        print(path.name)


if __name__ == "__main__":
    main()
