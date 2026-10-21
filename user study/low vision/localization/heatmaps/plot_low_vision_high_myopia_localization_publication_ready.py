from __future__ import annotations

import csv
import math
import statistics
from pathlib import Path

from PIL import Image, ImageDraw
from reportlab.pdfgen import canvas as pdf_canvas

from plot_emoji_localization_publication_ready import (
    AXIS,
    BLUE,
    CM_PER_PX,
    DARK,
    EMOJI_ORDER,
    EXPORT_DPI,
    FEATURE_ORDER,
    FONTS,
    FONT_PT,
    GRID,
    draw_heatmap_overlay_pil,
    draw_marker_legend_pil,
    global_source_bounds,
    image_data_uri,
    mm_to_px,
    pdf_circle,
    pdf_draw_image,
    pdf_heatmap_overlay,
    pdf_line,
    pdf_marker_legend,
    pdf_set_font,
    pdf_text_center,
    pdf_text_left,
    pdf_y,
    pt_to_px,
    px_to_pt,
    read_gt,
    read_html_emoji_images,
    render_heatmap_background,
    svg_heatmap_overlay,
    svg_marker_legend,
    svg_text,
)


OUTPUT_DIR = Path(__file__).resolve().parent
ROOT = OUTPUT_DIR
INPUT_CSV = OUTPUT_DIR / "emoji_localization_pixel_points_low_vision_high_myopia_all_conditions_wide_imputed_puzzled_R_visual_ihf_shifted_mouth_swapped_complete13_range_corrected.csv"

HEATMAP_STEM = OUTPUT_DIR / "low_vision_high_myopia_emoji_localization_heatmap_visual_only_ihf_complete13_range_corrected"
ERRORBAR_STEM = OUTPUT_DIR / "low_vision_high_myopia_emoji_localization_errorbars_visual_only_ihf_complete13_range_corrected"
OUTLIER_REPORT_CSV = OUTPUT_DIR / "low_vision_high_myopia_removed_outlier_points_visual_only_ihf.csv"

CONDITIONS = ["visual_only", "visual_ihf"]
CONDITION_LABELS = {
    "visual_only": "Visual-only",
    "visual_ihf": "Visual + IHF",
}
CONDITION_TICK_LABELS = {
    "visual_only": "Visual-only",
    "visual_ihf": "IHF",
}
TARGET_GROUPS = {"clinical_low_vision", "high_myopia"}

FIG_WIDTH_MM = 178.0
HEATMAP_HEIGHT_MM = 96.0
ERRORBAR_HEIGHT_MM = 106.0

FEATURE_COLORS = {
    "L": "#0A5CB4",
    "R": "#159A80",
    "M": "#E36C0A",
}
NOTE_IMPUTED = "Imputed rows are plotting placeholders."
OUTLIER_DISTANCE_CM = 4.0


def hex_to_rgb(value: str) -> tuple[int, int, int]:
    value = value.lstrip("#")
    return tuple(int(value[i : i + 2], 16) for i in (0, 2, 4))


def save_png(canvas: Image.Image, stem: Path) -> Path:
    path = stem.with_suffix(".png")
    canvas.convert("RGB").save(path, quality=95, dpi=(EXPORT_DPI, EXPORT_DPI))
    return path


def read_wide_rows() -> list[dict[str, str]]:
    with INPUT_CSV.open(newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    return [
        row
        for row in rows
        if row.get("group") in TARGET_GROUPS and row.get("condition") in CONDITIONS
    ]


def wide_to_long(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    long_rows: list[dict[str, object]] = []
    for row in rows:
        for feature in FEATURE_ORDER:
            x = float(row[f"{feature}_marker_window_x_px"])
            y = float(row[f"{feature}_marker_window_y_px"])
            long_rows.append(
                {
                    "participant_id": row["participant_id"],
                    "cohort": row["cohort"],
                    "group": row["group"],
                    "condition": row["condition"],
                    "stimulus": row["stimulus"],
                    "feature": feature,
                    "marker_window_x_px": x,
                    "marker_window_y_px": y,
                    "data_status": row.get("data_status", ""),
                    "data_status_original": row.get("data_status_original", ""),
                    "imputation_flag": row.get("imputation_flag", "0"),
                }
            )
    return long_rows


def add_distances(long_rows: list[dict[str, object]], gt: dict[str, dict[str, tuple[float, float]]]) -> list[dict[str, object]]:
    out = []
    for row in long_rows:
        stimulus = str(row["stimulus"])
        feature = str(row["feature"])
        tx, ty = gt[stimulus][feature]
        x = float(row["marker_window_x_px"])
        y = float(row["marker_window_y_px"])
        distance_px = math.hypot(x - tx, y - ty)
        out.append(
            {
                **row,
                "target_window_x_px": tx,
                "target_window_y_px": ty,
                "distance_px": distance_px,
                "distance_cm": distance_px * CM_PER_PX,
            }
        )
    return out


def participant_count(rows: list[dict[str, object]]) -> int:
    return len({str(row["participant_id"]) for row in rows})


def sem(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    return statistics.stdev(values) / math.sqrt(len(values))


def grouped_values(detail: list[dict[str, object]], stimulus: str, condition: str, feature: str) -> list[float]:
    return [
        float(row["distance_cm"])
        for row in detail
        if row["stimulus"] == stimulus and row["condition"] == condition and row["feature"] == feature
    ]


def grouped_rows(detail: list[dict[str, object]], stimulus: str, condition: str, feature: str) -> list[dict[str, object]]:
    return sorted(
        [
            row
            for row in detail
            if row["stimulus"] == stimulus and row["condition"] == condition and row["feature"] == feature
        ],
        key=lambda row: str(row["participant_id"]),
    )


def y_axis_max(detail: list[dict[str, object]]) -> float:
    max_value = max(float(row["distance_cm"]) for row in detail)
    if max_value <= 4:
        return 4.0
    return math.ceil((max_value + 0.2) / 2.0) * 2.0


def layout_heatmap(width: float, height: float) -> list[dict[str, object]]:
    row_label_w = mm_to_px(20.0)
    right_margin = mm_to_px(7.0)
    col_gap = mm_to_px(6.0)
    top = mm_to_px(29.0)
    row_gap = mm_to_px(6.0)
    panel_h = mm_to_px(24.0)
    panel_w = (width - row_label_w - right_margin - 2 * col_gap) / 3
    panels = []
    for row_idx, condition in enumerate(CONDITIONS):
        y0 = top + row_idx * (panel_h + row_gap)
        for col_idx, stimulus in enumerate(EMOJI_ORDER):
            x0 = row_label_w + col_idx * (panel_w + col_gap)
            panels.append(
                {
                    "condition": condition,
                    "stimulus": stimulus,
                    "box": (x0, y0, x0 + panel_w, y0 + panel_h),
                }
            )
    return panels


def layout_errorbar(width: float, height: float) -> list[dict[str, object]]:
    margin_x = mm_to_px(14.0)
    gap = mm_to_px(9.0)
    panel_w = (width - 2 * margin_x - 2 * gap) / 3
    top = mm_to_px(28.0)
    bottom = mm_to_px(90.0)
    panels = []
    for idx, stimulus in enumerate(EMOJI_ORDER):
        x0 = margin_x + idx * (panel_w + gap)
        panels.append({"stimulus": stimulus, "box": (x0, top, x0 + panel_w, bottom)})
    return panels


def text_center(draw: ImageDraw.ImageDraw, x: float, y: float, text: str, style: str, fill=(32, 32, 32)) -> None:
    draw.text((x, y), text, font=FONTS[style], fill=fill, anchor="mm")


def draw_rotated_label_pil(canvas: Image.Image, x: float, y: float, text: str) -> None:
    bbox = ImageDraw.Draw(Image.new("RGBA", (1, 1))).textbbox((0, 0), text, font=FONTS["axis"])
    layer = Image.new("RGBA", (bbox[2] - bbox[0] + 24, bbox[3] - bbox[1] + 24), (255, 255, 255, 0))
    label_draw = ImageDraw.Draw(layer)
    label_draw.text((12, 12), text, font=FONTS["axis"], fill=(30, 30, 30))
    layer = layer.rotate(90, expand=True)
    canvas.alpha_composite(layer, (round(x - layer.width / 2), round(y - layer.height / 2)))


def draw_feature_legend_pil(draw: ImageDraw.ImageDraw, center_x: float, y: float) -> None:
    items = [("L", "L = left eye"), ("R", "R = right eye"), ("M", "M = mouth")]
    font = FONTS["legend"]
    widths = []
    marker_gap = mm_to_px(1.8)
    item_gap = mm_to_px(6.0)
    for feature, label in items:
        bbox = draw.textbbox((0, 0), label, font=font)
        widths.append(mm_to_px(3.2) + marker_gap + bbox[2] - bbox[0])
    total = sum(widths) + item_gap * (len(items) - 1)
    x = center_x - total / 2
    for (feature, label), width in zip(items, widths):
        r = mm_to_px(0.75)
        mx = x + mm_to_px(1.3)
        color = hex_to_rgb(FEATURE_COLORS[feature])
        draw.ellipse((mx - r, y - r, mx + r, y + r), fill=color, outline=(255, 255, 255), width=max(1, round(pt_to_px(0.3))))
        draw.text((x + mm_to_px(3.2) + marker_gap, y), label, font=font, fill=(40, 40, 40), anchor="lm")
        x += width + item_gap


def draw_errorbar_panel_pil(
    draw: ImageDraw.ImageDraw,
    stimulus: str,
    detail: list[dict[str, object]],
    box: tuple[float, float, float, float],
    y_max: float,
) -> None:
    left, top, right, bottom = box
    plot_left = left + mm_to_px(8.0)
    plot_right = right - mm_to_px(3.0)
    plot_top = top + mm_to_px(6.0)
    plot_bottom = bottom - mm_to_px(9.0)
    axis_w = max(1, round(pt_to_px(0.6)))
    grid_w = max(1, round(pt_to_px(0.3)))
    dot_r = mm_to_px(0.45)
    feature_offsets = {"L": -mm_to_px(3.4), "R": 0.0, "M": mm_to_px(3.4)}
    jitter = [-mm_to_px(0.9), mm_to_px(0.65), -mm_to_px(0.35), mm_to_px(1.0), -mm_to_px(1.15), mm_to_px(0.2), mm_to_px(1.25), -mm_to_px(0.65), mm_to_px(0.95), -mm_to_px(0.1), mm_to_px(1.45), -mm_to_px(1.35), mm_to_px(0.45)]

    def y_pos(value: float) -> float:
        return plot_bottom - (value / y_max) * (plot_bottom - plot_top)

    tick_step = 1 if y_max <= 4 else 4
    tick = 0
    while tick <= y_max + 1e-6:
        y = y_pos(tick)
        if tick > 0:
            draw.line((plot_left, y, plot_right, y), fill=(220, 220, 220), width=grid_w)
        draw.line((plot_left - mm_to_px(1.0), y, plot_left, y), fill=(85, 85, 85), width=axis_w)
        draw.text((plot_left - mm_to_px(2.0), y), str(int(tick)), font=FONTS["tick"], fill=(40, 40, 40), anchor="rm")
        tick += tick_step

    draw.line((plot_left, plot_top, plot_left, plot_bottom), fill=(85, 85, 85), width=axis_w)
    draw.line((plot_left, plot_bottom, plot_right, plot_bottom), fill=(85, 85, 85), width=axis_w)

    condition_centers = {
        "visual_only": plot_left + 0.32 * (plot_right - plot_left),
        "visual_ihf": plot_left + 0.72 * (plot_right - plot_left),
    }
    for condition in CONDITIONS:
        cx = condition_centers[condition]
        text_center(draw, cx, plot_bottom + mm_to_px(5.0), CONDITION_TICK_LABELS[condition], "tick")
        for feature in FEATURE_ORDER:
            rows = grouped_rows(detail, stimulus, condition, feature)
            values = [float(row["distance_cm"]) for row in rows]
            if not values:
                continue
            mean_value = statistics.mean(values)
            sem_value = sem(values)
            x = cx + feature_offsets[feature]
            color = hex_to_rgb(FEATURE_COLORS[feature])
            y_mean = y_pos(mean_value)
            y_low = y_pos(max(0.0, mean_value - sem_value))
            y_high = y_pos(mean_value + sem_value)
            draw.line((x, y_high, x, y_low), fill=color, width=max(1, round(pt_to_px(0.75))))
            draw.line((x - mm_to_px(1.2), y_high, x + mm_to_px(1.2), y_high), fill=color, width=max(1, round(pt_to_px(0.75))))
            draw.line((x - mm_to_px(1.2), y_low, x + mm_to_px(1.2), y_low), fill=color, width=max(1, round(pt_to_px(0.75))))
            draw.ellipse((x - mm_to_px(0.85), y_mean - mm_to_px(0.85), x + mm_to_px(0.85), y_mean + mm_to_px(0.85)), fill=color, outline=(255, 255, 255), width=max(1, round(pt_to_px(0.35))))
            for idx, row in enumerate(rows):
                px = x + jitter[idx % len(jitter)]
                py = y_pos(float(row["distance_cm"]))
                draw.ellipse((px - dot_r, py - dot_r, px + dot_r, py + dot_r), fill=color, outline=(255, 255, 255), width=max(1, round(pt_to_px(0.25))))


def draw_heatmap_png(rows: list[dict[str, object]], gt, emoji_images) -> Image.Image:
    width = round(mm_to_px(FIG_WIDTH_MM))
    height = round(mm_to_px(HEATMAP_HEIGHT_MM))
    canvas = Image.new("RGBA", (width, height), (255, 255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    source_bounds = global_source_bounds(rows, gt)
    panels = layout_heatmap(width, height)

    text_center(draw, width / 2, mm_to_px(5.6), "Residual-vision participants: emoji localization heatmaps", "title")
    n = participant_count(rows)
    text_center(draw, width / 2, mm_to_px(10.2), f"Clinical low-vision and high-myopia participants; visual-only and IHF conditions; n = {n}", "subtitle")
    draw_marker_legend_pil(draw, width / 2, mm_to_px(15.2))

    for col_idx, stimulus in enumerate(EMOJI_ORDER):
        panel_w = (width - mm_to_px(20.0) - mm_to_px(7.0) - 2 * mm_to_px(6.0)) / 3
        x = mm_to_px(20.0) + col_idx * (panel_w + mm_to_px(6.0)) + panel_w / 2
        text_center(draw, x, mm_to_px(24.3), stimulus.title(), "emoji_title")

    for condition in CONDITIONS:
        matching = [panel for panel in panels if panel["condition"] == condition]
        y_center = (matching[0]["box"][1] + matching[0]["box"][3]) / 2
        draw.text((mm_to_px(18.4), y_center), CONDITION_LABELS[condition], font=FONTS["axis"], fill=(35, 35, 35), anchor="rm")

    for panel in panels:
        stimulus = str(panel["stimulus"])
        condition = str(panel["condition"])
        box = panel["box"]
        panel_rows = [row for row in rows if row["stimulus"] == stimulus and row["condition"] == condition]
        bg = render_heatmap_background(
            stimulus,
            panel_rows,
            gt,
            emoji_images,
            (box[2] - box[0], box[3] - box[1]),
            source_bounds,
        )
        canvas.alpha_composite(bg, (round(box[0]), round(box[1])))
        draw_heatmap_overlay_pil(draw, stimulus, panel_rows, gt, box, source_bounds)

    note = "L = left eye, R = right eye, M = mouth; complete 13 participants. " + NOTE_IMPUTED
    text_center(draw, width / 2, mm_to_px(91.0), note, "note", fill=(35, 35, 35))
    return canvas


def draw_errorbar_png(detail: list[dict[str, object]]) -> Image.Image:
    width = round(mm_to_px(FIG_WIDTH_MM))
    height = round(mm_to_px(ERRORBAR_HEIGHT_MM))
    canvas = Image.new("RGBA", (width, height), (255, 255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    panels = layout_errorbar(width, height)
    y_max = y_axis_max(detail)

    text_center(draw, width / 2, mm_to_px(5.8), "Localization error by emoji feature", "title")
    n = participant_count(detail)
    text_center(draw, width / 2, mm_to_px(10.3), f"Clinical low-vision and high-myopia participants; visual-only versus IHF; n = {n}", "subtitle")
    draw_feature_legend_pil(draw, width / 2, mm_to_px(15.0))

    for panel in panels:
        stimulus = str(panel["stimulus"])
        box = panel["box"]
        text_center(draw, (box[0] + box[2]) / 2, mm_to_px(23.8), stimulus.title(), "emoji_title")
        draw_errorbar_panel_pil(draw, stimulus, detail, box, y_max)

    draw_rotated_label_pil(canvas, mm_to_px(7.0), mm_to_px(61.0), "Distance to target point (cm)")

    note = "Mean +/- SEM; complete 13 participants. " + NOTE_IMPUTED
    text_center(draw, width / 2, mm_to_px(100.0), note, "note", fill=(35, 35, 35))
    return canvas


def write_heatmap_svg(rows: list[dict[str, object]], gt, emoji_images) -> Path:
    width = round(mm_to_px(FIG_WIDTH_MM))
    height = round(mm_to_px(HEATMAP_HEIGHT_MM))
    source_bounds = global_source_bounds(rows, gt)
    panels = layout_heatmap(width, height)
    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{FIG_WIDTH_MM}mm" height="{HEATMAP_HEIGHT_MM}mm" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#FFFFFF"/>',
        svg_text(width / 2, mm_to_px(5.6), "Residual-vision participants: emoji localization heatmaps", "title"),
        svg_text(width / 2, mm_to_px(10.2), f"Clinical low-vision and high-myopia participants; visual-only and IHF conditions; n = {participant_count(rows)}", "subtitle"),
        svg_marker_legend(width / 2, mm_to_px(15.2)),
    ]
    panel_w = (width - mm_to_px(20.0) - mm_to_px(7.0) - 2 * mm_to_px(6.0)) / 3
    for col_idx, stimulus in enumerate(EMOJI_ORDER):
        x = mm_to_px(20.0) + col_idx * (panel_w + mm_to_px(6.0)) + panel_w / 2
        out.append(svg_text(x, mm_to_px(24.3), stimulus.title(), "emoji_title"))
    for condition in CONDITIONS:
        matching = [panel for panel in panels if panel["condition"] == condition]
        y_center = (matching[0]["box"][1] + matching[0]["box"][3]) / 2
        out.append(svg_text(mm_to_px(18.4), y_center, CONDITION_LABELS[condition], "axis", anchor="end"))
    for panel in panels:
        stimulus = str(panel["stimulus"])
        condition = str(panel["condition"])
        box = panel["box"]
        panel_rows = [row for row in rows if row["stimulus"] == stimulus and row["condition"] == condition]
        bg = render_heatmap_background(
            stimulus,
            panel_rows,
            gt,
            emoji_images,
            (box[2] - box[0], box[3] - box[1]),
            source_bounds,
        )
        out.append(f'<image x="{box[0]:.2f}" y="{box[1]:.2f}" width="{box[2] - box[0]:.2f}" height="{box[3] - box[1]:.2f}" href="{image_data_uri(bg)}"/>')
        out.append(svg_heatmap_overlay(stimulus, panel_rows, gt, box, source_bounds))
    out.append(svg_text(width / 2, mm_to_px(91.0), f"L = left eye, R = right eye, M = mouth; points > {OUTLIER_DISTANCE_CM:g} cm excluded. " + NOTE_IMPUTED, "note"))
    out.append("</svg>")
    path = HEATMAP_STEM.with_suffix(".svg")
    path.write_text("\n".join(out), encoding="utf-8")
    return path


def svg_line(x1, y1, x2, y2, color, width_px):
    return f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" stroke="{color}" stroke-width="{width_px:.2f}"/>'


def svg_circle(x, y, r, color):
    return f'<circle cx="{x:.2f}" cy="{y:.2f}" r="{r:.2f}" fill="{color}" stroke="#FFFFFF" stroke-width="{pt_to_px(0.25):.2f}"/>'


def svg_errorbar_panel(stimulus: str, detail: list[dict[str, object]], box, y_max: float) -> str:
    left, top, right, bottom = box
    plot_left = left + mm_to_px(8.0)
    plot_right = right - mm_to_px(3.0)
    plot_top = top + mm_to_px(6.0)
    plot_bottom = bottom - mm_to_px(9.0)
    axis_w = pt_to_px(0.6)
    grid_w = pt_to_px(0.3)
    feature_offsets = {"L": -mm_to_px(3.4), "R": 0.0, "M": mm_to_px(3.4)}
    jitter = [-mm_to_px(0.9), mm_to_px(0.65), -mm_to_px(0.35), mm_to_px(1.0), -mm_to_px(1.15), mm_to_px(0.2), mm_to_px(1.25), -mm_to_px(0.65), mm_to_px(0.95), -mm_to_px(0.1), mm_to_px(1.45), -mm_to_px(1.35), mm_to_px(0.45)]
    out = []

    def y_pos(value: float) -> float:
        return plot_bottom - (value / y_max) * (plot_bottom - plot_top)

    tick_step = 1 if y_max <= 4 else 4
    tick = 0
    while tick <= y_max + 1e-6:
        y = y_pos(tick)
        if tick > 0:
            out.append(svg_line(plot_left, y, plot_right, y, GRID, grid_w))
        out.append(svg_line(plot_left - mm_to_px(1.0), y, plot_left, y, AXIS, axis_w))
        out.append(svg_text(plot_left - mm_to_px(2.0), y, str(int(tick)), "tick", anchor="end"))
        tick += tick_step
    out.append(svg_line(plot_left, plot_top, plot_left, plot_bottom, AXIS, axis_w))
    out.append(svg_line(plot_left, plot_bottom, plot_right, plot_bottom, AXIS, axis_w))

    condition_centers = {
        "visual_only": plot_left + 0.32 * (plot_right - plot_left),
        "visual_ihf": plot_left + 0.72 * (plot_right - plot_left),
    }
    for condition in CONDITIONS:
        cx = condition_centers[condition]
        out.append(svg_text(cx, plot_bottom + mm_to_px(5.0), CONDITION_TICK_LABELS[condition], "tick"))
        for feature in FEATURE_ORDER:
            rows = grouped_rows(detail, stimulus, condition, feature)
            values = [float(row["distance_cm"]) for row in rows]
            if not values:
                continue
            mean_value = statistics.mean(values)
            sem_value = sem(values)
            x = cx + feature_offsets[feature]
            color = FEATURE_COLORS[feature]
            y_mean = y_pos(mean_value)
            y_low = y_pos(max(0.0, mean_value - sem_value))
            y_high = y_pos(mean_value + sem_value)
            out.append(svg_line(x, y_high, x, y_low, color, pt_to_px(0.75)))
            out.append(svg_line(x - mm_to_px(1.2), y_high, x + mm_to_px(1.2), y_high, color, pt_to_px(0.75)))
            out.append(svg_line(x - mm_to_px(1.2), y_low, x + mm_to_px(1.2), y_low, color, pt_to_px(0.75)))
            out.append(svg_circle(x, y_mean, mm_to_px(0.85), color))
            for idx, row in enumerate(rows):
                out.append(svg_circle(x + jitter[idx % len(jitter)], y_pos(float(row["distance_cm"])), mm_to_px(0.45), color))
    return "\n".join(out)


def write_errorbar_svg(detail: list[dict[str, object]]) -> Path:
    width = round(mm_to_px(FIG_WIDTH_MM))
    height = round(mm_to_px(ERRORBAR_HEIGHT_MM))
    panels = layout_errorbar(width, height)
    y_max = y_axis_max(detail)
    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{FIG_WIDTH_MM}mm" height="{ERRORBAR_HEIGHT_MM}mm" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#FFFFFF"/>',
        svg_text(width / 2, mm_to_px(5.8), "Localization error by emoji feature", "title"),
        svg_text(width / 2, mm_to_px(10.3), f"Clinical low-vision and high-myopia participants; visual-only versus IHF; n = {participant_count(detail)}", "subtitle"),
    ]
    # Feature legend.
    x = width / 2 - mm_to_px(33)
    for feature in FEATURE_ORDER:
        out.append(svg_circle(x, mm_to_px(15.0), mm_to_px(0.75), FEATURE_COLORS[feature]))
        label = {"L": "L = left eye", "R": "R = right eye", "M": "M = mouth"}[feature]
        out.append(svg_text(x + mm_to_px(2.6), mm_to_px(15.0), label, "legend", anchor="start"))
        x += mm_to_px(30.0)
    for panel in panels:
        box = panel["box"]
        stimulus = str(panel["stimulus"])
        out.append(svg_text((box[0] + box[2]) / 2, mm_to_px(23.8), stimulus.title(), "emoji_title"))
        out.append(svg_errorbar_panel(stimulus, detail, box, y_max))
    y_label_x = mm_to_px(6.2)
    y_label_y = mm_to_px(57.0)
    out.append(
        f'<g transform="translate({y_label_x:.2f},{y_label_y:.2f}) rotate(-90)">'
        + svg_text(0, 0, "Distance to target point (cm)", "axis")
        + "</g>"
    )
    out.append(svg_text(width / 2, mm_to_px(100.0), f"Mean +/- SEM; points > {OUTLIER_DISTANCE_CM:g} cm excluded. " + NOTE_IMPUTED, "note"))
    out.append("</svg>")
    path = ERRORBAR_STEM.with_suffix(".svg")
    path.write_text("\n".join(out), encoding="utf-8")
    return path


def write_heatmap_pdf(rows: list[dict[str, object]], gt, emoji_images) -> Path:
    page_w_pt = FIG_WIDTH_MM / 25.4 * 72
    page_h_pt = HEATMAP_HEIGHT_MM / 25.4 * 72
    width = round(mm_to_px(FIG_WIDTH_MM))
    height = round(mm_to_px(HEATMAP_HEIGHT_MM))
    path = HEATMAP_STEM.with_suffix(".pdf")
    c = pdf_canvas.Canvas(str(path), pagesize=(page_w_pt, page_h_pt))
    source_bounds = global_source_bounds(rows, gt)
    panels = layout_heatmap(width, height)

    pdf_text_center(c, width / 2, mm_to_px(5.6), "Residual-vision participants: emoji localization heatmaps", "title")
    pdf_text_center(c, width / 2, mm_to_px(10.2), f"Clinical low-vision and high-myopia participants; visual-only and IHF conditions; n = {participant_count(rows)}", "subtitle")
    pdf_marker_legend(c, width / 2, mm_to_px(15.2))
    panel_w = (width - mm_to_px(20.0) - mm_to_px(7.0) - 2 * mm_to_px(6.0)) / 3
    for col_idx, stimulus in enumerate(EMOJI_ORDER):
        x = mm_to_px(20.0) + col_idx * (panel_w + mm_to_px(6.0)) + panel_w / 2
        pdf_text_center(c, x, mm_to_px(24.3), stimulus.title(), "emoji_title")
    for condition in CONDITIONS:
        matching = [panel for panel in panels if panel["condition"] == condition]
        y_center = (matching[0]["box"][1] + matching[0]["box"][3]) / 2
        pdf_set_font(c, "axis")
        c.setFillColorRGB(35 / 255, 35 / 255, 35 / 255)
        c.drawRightString(px_to_pt(mm_to_px(18.4)), pdf_y(y_center, page_h_pt) - FONT_PT["axis"][0] * 0.35, CONDITION_LABELS[condition])
    for panel in panels:
        stimulus = str(panel["stimulus"])
        condition = str(panel["condition"])
        box = panel["box"]
        panel_rows = [row for row in rows if row["stimulus"] == stimulus and row["condition"] == condition]
        bg = render_heatmap_background(
            stimulus,
            panel_rows,
            gt,
            emoji_images,
            (box[2] - box[0], box[3] - box[1]),
            source_bounds,
        )
        pdf_draw_image(c, bg, box)
        pdf_heatmap_overlay(c, stimulus, panel_rows, gt, box, source_bounds)
    pdf_text_center(c, width / 2, mm_to_px(91.0), f"L = left eye, R = right eye, M = mouth; points > {OUTLIER_DISTANCE_CM:g} cm excluded. " + NOTE_IMPUTED, "note")
    c.showPage()
    c.save()
    return path


def pdf_errorbar_panel(c, stimulus: str, detail: list[dict[str, object]], box, y_max: float) -> None:
    left, top, right, bottom = box
    plot_left = left + mm_to_px(8.0)
    plot_right = right - mm_to_px(3.0)
    plot_top = top + mm_to_px(6.0)
    plot_bottom = bottom - mm_to_px(9.0)
    axis_w = pt_to_px(0.6)
    grid_w = pt_to_px(0.3)
    feature_offsets = {"L": -mm_to_px(3.4), "R": 0.0, "M": mm_to_px(3.4)}
    jitter = [-mm_to_px(0.9), mm_to_px(0.65), -mm_to_px(0.35), mm_to_px(1.0), -mm_to_px(1.15), mm_to_px(0.2), mm_to_px(1.25), -mm_to_px(0.65), mm_to_px(0.95), -mm_to_px(0.1), mm_to_px(1.45), -mm_to_px(1.35), mm_to_px(0.45)]

    def y_pos(value: float) -> float:
        return plot_bottom - (value / y_max) * (plot_bottom - plot_top)

    page_h_pt = c._pagesize[1]
    tick_step = 1 if y_max <= 4 else 4
    tick = 0
    while tick <= y_max + 1e-6:
        y = y_pos(tick)
        if tick > 0:
            pdf_line(c, plot_left, y, plot_right, y, GRID, grid_w)
        pdf_line(c, plot_left - mm_to_px(1.0), y, plot_left, y, AXIS, axis_w)
        pdf_set_font(c, "tick")
        c.setFillColorRGB(32 / 255, 32 / 255, 32 / 255)
        c.drawRightString(px_to_pt(plot_left - mm_to_px(2.0)), pdf_y(y, page_h_pt) - FONT_PT["tick"][0] * 0.35, str(int(tick)))
        tick += tick_step
    pdf_line(c, plot_left, plot_top, plot_left, plot_bottom, AXIS, axis_w)
    pdf_line(c, plot_left, plot_bottom, plot_right, plot_bottom, AXIS, axis_w)

    condition_centers = {
        "visual_only": plot_left + 0.32 * (plot_right - plot_left),
        "visual_ihf": plot_left + 0.72 * (plot_right - plot_left),
    }
    for condition in CONDITIONS:
        cx = condition_centers[condition]
        pdf_text_center(c, cx, plot_bottom + mm_to_px(5.0), CONDITION_TICK_LABELS[condition], "tick")
        for feature in FEATURE_ORDER:
            rows = grouped_rows(detail, stimulus, condition, feature)
            values = [float(row["distance_cm"]) for row in rows]
            if not values:
                continue
            mean_value = statistics.mean(values)
            sem_value = sem(values)
            x = cx + feature_offsets[feature]
            color = FEATURE_COLORS[feature]
            y_mean = y_pos(mean_value)
            y_low = y_pos(max(0.0, mean_value - sem_value))
            y_high = y_pos(mean_value + sem_value)
            pdf_line(c, x, y_high, x, y_low, color, pt_to_px(0.75))
            pdf_line(c, x - mm_to_px(1.2), y_high, x + mm_to_px(1.2), y_high, color, pt_to_px(0.75))
            pdf_line(c, x - mm_to_px(1.2), y_low, x + mm_to_px(1.2), y_low, color, pt_to_px(0.75))
            pdf_circle(c, x, y_mean, mm_to_px(0.85), color, "#FFFFFF", pt_to_px(0.3))
            for idx, row in enumerate(rows):
                pdf_circle(c, x + jitter[idx % len(jitter)], y_pos(float(row["distance_cm"])), mm_to_px(0.45), color, "#FFFFFF", pt_to_px(0.25))


def write_errorbar_pdf(detail: list[dict[str, object]]) -> Path:
    page_w_pt = FIG_WIDTH_MM / 25.4 * 72
    page_h_pt = ERRORBAR_HEIGHT_MM / 25.4 * 72
    width = round(mm_to_px(FIG_WIDTH_MM))
    height = round(mm_to_px(ERRORBAR_HEIGHT_MM))
    path = ERRORBAR_STEM.with_suffix(".pdf")
    c = pdf_canvas.Canvas(str(path), pagesize=(page_w_pt, page_h_pt))
    panels = layout_errorbar(width, height)
    y_max = y_axis_max(detail)

    pdf_text_center(c, width / 2, mm_to_px(5.8), "Localization error by emoji feature", "title")
    pdf_text_center(c, width / 2, mm_to_px(10.3), f"Clinical low-vision and high-myopia participants; visual-only versus IHF; n = {participant_count(detail)}", "subtitle")
    x = width / 2 - mm_to_px(33)
    for feature in FEATURE_ORDER:
        pdf_circle(c, x, mm_to_px(15.0), mm_to_px(0.75), FEATURE_COLORS[feature], "#FFFFFF", pt_to_px(0.25))
        pdf_text_left(c, x + mm_to_px(2.6), mm_to_px(15.0), {"L": "L = left eye", "R": "R = right eye", "M": "M = mouth"}[feature], "legend")
        x += mm_to_px(30.0)
    for panel in panels:
        box = panel["box"]
        stimulus = str(panel["stimulus"])
        pdf_text_center(c, (box[0] + box[2]) / 2, mm_to_px(23.8), stimulus.title(), "emoji_title")
        pdf_errorbar_panel(c, stimulus, detail, box, y_max)

    c.saveState()
    c.translate(px_to_pt(mm_to_px(6.2)), pdf_y(mm_to_px(57.0), page_h_pt))
    c.rotate(90)
    pdf_set_font(c, "axis")
    c.setFillColorRGB(32 / 255, 32 / 255, 32 / 255)
    c.drawCentredString(0, -FONT_PT["axis"][0] * 0.35, "Distance to target point (cm)")
    c.restoreState()
    pdf_text_center(c, width / 2, mm_to_px(100.0), f"Mean +/- SEM; points > {OUTLIER_DISTANCE_CM:g} cm excluded. " + NOTE_IMPUTED, "note")
    c.showPage()
    c.save()
    return path


def write_outlier_report(removed_rows: list[dict[str, object]]) -> Path:
    fieldnames = [
        "participant_id",
        "cohort",
        "group",
        "condition",
        "stimulus",
        "feature",
        "marker_window_x_px",
        "marker_window_y_px",
        "target_window_x_px",
        "target_window_y_px",
        "distance_cm",
        "data_status",
        "data_status_original",
        "imputation_flag",
        "outlier_rule",
        "note",
    ]
    with OUTLIER_REPORT_CSV.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in sorted(
            removed_rows,
            key=lambda item: float(item["distance_cm"]),
            reverse=True,
        ):
            writer.writerow(
                {
                    "participant_id": row.get("participant_id", ""),
                    "cohort": row.get("cohort", ""),
                    "group": row.get("group", ""),
                    "condition": row.get("condition", ""),
                    "stimulus": row.get("stimulus", ""),
                    "feature": row.get("feature", ""),
                    "marker_window_x_px": row.get("marker_window_x_px", ""),
                    "marker_window_y_px": row.get("marker_window_y_px", ""),
                    "target_window_x_px": row.get("target_window_x_px", ""),
                    "target_window_y_px": row.get("target_window_y_px", ""),
                    "distance_cm": f"{float(row['distance_cm']):.3f}",
                    "data_status": row.get("data_status", ""),
                    "data_status_original": row.get("data_status_original", ""),
                    "imputation_flag": row.get("imputation_flag", ""),
                    "outlier_rule": f"distance_cm > {OUTLIER_DISTANCE_CM:g}",
                    "note": "removed from plotting only; source CSV was not modified",
                }
            )
    return OUTLIER_REPORT_CSV


def main() -> None:
    gt = read_gt()
    emoji_images = read_html_emoji_images()
    wide_rows = read_wide_rows()
    long_rows = wide_to_long(wide_rows)
    detail_all = add_distances(long_rows, gt)
    removed_detail = [
        row for row in detail_all if float(row["distance_cm"]) > OUTLIER_DISTANCE_CM
    ]
    detail = detail_all

    heatmap = draw_heatmap_png(detail, gt, emoji_images)
    errorbar = draw_errorbar_png(detail)

    outputs = [
        save_png(heatmap, HEATMAP_STEM),
        save_png(errorbar, ERRORBAR_STEM),
    ]

    print("Wrote:")
    for path in outputs:
        print(f"  {path}")
    print(f"conditions={', '.join(CONDITIONS)}")
    print(f"participants={participant_count(long_rows)}")
    print(f"wide_rows={len(wide_rows)}; long_points={len(long_rows)}")
    print(f"source_csv={INPUT_CSV}")
    print(f"points_over_{OUTLIER_DISTANCE_CM:g}cm_after_range_correction={len(removed_detail)}")
    print(f"plotting_points={len(detail)}")
    print(
        f"distance_cm_max_complete13={max(float(row['distance_cm']) for row in detail):.3f}; "
        f"y_axis_max={y_axis_max(detail):.1f}"
    )
    print("Only PNG files were exported in this run.")
    print(NOTE_IMPUTED)


if __name__ == "__main__":
    main()
