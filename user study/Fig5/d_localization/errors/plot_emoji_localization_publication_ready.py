from pathlib import Path
import base64
import csv
import io
import math
import statistics
from xml.sax.saxutils import escape

from PIL import Image, ImageDraw, ImageFont, ImageFilter
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas as pdf_canvas


OUTPUT_DIR = Path(__file__).resolve().parent
ROOT = OUTPUT_DIR
PARTICIPANT_CSV = OUTPUT_DIR / "emoji_localization_window_px_by_participant_visual_ihf_corrected_puzzled_R_adjusted_p005_recovered.csv"
GT_CSV = OUTPUT_DIR / "emoji_ground_truth_window_px_puzzled_eye_adjusted.csv"
HTML_SOURCE = ROOT / "reduced_vision_social_feed_demo_ihf_posture_redmi_standalone.html"

FIG5C_STEM = OUTPUT_DIR / "emoji_localization_fig5c_blind_visual_ihf_window_percent"
HEATMAP_STEM = OUTPUT_DIR / "blind_emoji_lrm_point_distribution_heatmap"

EMOJI_ORDER = ["puzzled", "screaming", "impatient"]
FEATURE_ORDER = ["L", "R", "M"]
HTML_ALT_TO_STIMULUS = {
    "confounded emoji": "puzzled",
    "screaming emoji": "screaming",
    "eye-roll emoji": "impatient",
}
IMAGE_ANCHORS = {
    "puzzled": {"L": (0.36, 0.49), "R": (0.65, 0.49), "M": (0.50, 0.70)},
    "screaming": {"L": (0.33, 0.46), "R": (0.67, 0.46), "M": (0.50, 0.72)},
    "impatient": {"L": (0.35, 0.43), "R": (0.69, 0.43), "M": (0.50, 0.71)},
}

EMOJI_WINDOW_SIZE_PX = 410.0
PX_PER_2_2_CM = 1400 - 1269
CM_PER_PX = 2.2 / PX_PER_2_2_CM
LOCALIZATION_WINDOW_WIDTH_PX = 665.0
LOCALIZATION_WINDOW_HEIGHT_PX = 440.0
LOCALIZATION_WINDOW_DIAGONAL_PX = math.hypot(
    LOCALIZATION_WINDOW_WIDTH_PX,
    LOCALIZATION_WINDOW_HEIGHT_PX,
)
ERROR_Y_MAX_PERCENT =  16.0
ERROR_Y_TICKS_PERCENT = (0, 4, 8, 12, 16)

EXPORT_DPI = 600
FIG_WIDTH_MM = 178.0
FIG5C_HEIGHT_MM = 115.0
HEATMAP_HEIGHT_MM = 62.0

BLUE = "#0A5CB4"
RED = "#DC1E32"
DARK = "#202020"
GRID = "#DCDCDC"
AXIS = "#555555"
BOX_FILL = "#B7D5F0"


def mm_to_px(mm):
    return mm / 25.4 * EXPORT_DPI


def pt_to_px(pt):
    return pt / 72.0 * EXPORT_DPI


def px_to_pt(px):
    return px / EXPORT_DPI * 72.0


def font_pt(size_pt, bold=False):
    size_px = max(1, round(pt_to_px(size_pt)))
    candidates = [
        r"C:\Windows\Fonts\arialbd.ttf" if bold else r"C:\Windows\Fonts\arial.ttf",
        r"C:\Windows\Fonts\helvetica.ttf",
        r"C:\Windows\Fonts\calibrib.ttf" if bold else r"C:\Windows\Fonts\calibri.ttf",
    ]
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size_px)
        except OSError:
            pass
    return ImageFont.load_default()


FONTS = {
    "panel": font_pt(13, True),
    "title": font_pt(12.5, True),
    "subtitle": font_pt(8, False),
    "legend": font_pt(7.8, False),
    "emoji_title": font_pt(9, True),
    "axis": font_pt(8, False),
    "tick": font_pt(7, False),
    "note": font_pt(7, False),
    "feature": font_pt(7.5, False),
}

FONT_PT = {
    "panel": (13, True),
    "title": (12.5, True),
    "subtitle": (8, False),
    "legend": (7.8, False),
    "emoji_title": (9, True),
    "axis": (8, False),
    "tick": (7, False),
    "note": (7, False),
    "feature": (7.5, False),
}


def read_gt():
    gt = {}
    with GT_CSV.open(newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            stimulus = row["stimulus"]
            gt[stimulus] = {}
            for feature in FEATURE_ORDER:
                gt[stimulus][feature] = (
                    float(row[f"{feature}_target_window_x_px"]),
                    float(row[f"{feature}_target_window_y_px"]),
                )
    return gt


def read_marker_rows():
    rows = []
    with PARTICIPANT_CSV.open(newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            if row["group"] != "blind" or row["condition"] != "visual_ihf":
                continue
            stimulus = row["stimulus"]
            if stimulus not in EMOJI_ORDER:
                continue
            for feature in FEATURE_ORDER:
                x_value = row.get(f"{feature}_marker_window_x_px", "").strip()
                y_value = row.get(f"{feature}_marker_window_y_px", "").strip()
                if not x_value or not y_value:
                    continue
                rows.append(
                    {
                        "participant_id": row["participant_id"],
                        "cohort": row["cohort"],
                        "condition": row["condition"],
                        "stimulus": stimulus,
                        "feature": feature,
                        "marker_window_x_px": float(x_value),
                        "marker_window_y_px": float(y_value),
                    }
                )
    return rows


def read_html_emoji_images():
    import re

    html = HTML_SOURCE.read_text(encoding="utf-8", errors="ignore")
    matches = re.findall(
        r'<div class="emoji"><img src="data:image/png;base64,([^"]+)" alt="([^"]+)"',
        html,
    )
    images = {}
    for encoded, alt in matches:
        stimulus = HTML_ALT_TO_STIMULUS.get(alt)
        if not stimulus:
            continue
        images[stimulus] = Image.open(io.BytesIO(base64.b64decode(encoded))).convert("RGBA")
    missing = [stimulus for stimulus in EMOJI_ORDER if stimulus not in images]
    if missing:
        raise ValueError(f"Missing HTML emoji image(s): {missing}")
    return images


def compute_distances(rows, gt):
    detail = []
    for row in rows:
        tx, ty = gt[row["stimulus"]][row["feature"]]
        dx = row["marker_window_x_px"] - tx
        dy = row["marker_window_y_px"] - ty
        distance_px = math.hypot(dx, dy)
        detail.append(
            {
                **row,
                "target_window_x_px": tx,
                "target_window_y_px": ty,
                "delta_x_px": dx,
                "delta_y_px": dy,
                "distance_px": distance_px,
                "distance_cm": distance_px * CM_PER_PX,
                "distance_percent": distance_px / LOCALIZATION_WINDOW_DIAGONAL_PX * 100.0,
            }
        )
    return detail


def quantile(values, q):
    ordered = sorted(values)
    if not ordered:
        return 0.0
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return ordered[lo]
    return ordered[lo] * (hi - pos) + ordered[hi] * (pos - lo)


def box_stats(values):
    return {
        "min": min(values),
        "q1": quantile(values, 0.25),
        "median": statistics.median(values),
        "q3": quantile(values, 0.75),
        "max": max(values),
    }


def image_window_rect(stimulus, gt):
    anchors = IMAGE_ANCHORS[stimulus]
    anchor_center = (
        sum(anchors[feature][0] for feature in FEATURE_ORDER) / len(FEATURE_ORDER),
        sum(anchors[feature][1] for feature in FEATURE_ORDER) / len(FEATURE_ORDER),
    )
    dst_center = (
        sum(gt[stimulus][feature][0] for feature in FEATURE_ORDER) / len(FEATURE_ORDER),
        sum(gt[stimulus][feature][1] for feature in FEATURE_ORDER) / len(FEATURE_ORDER),
    )
    left = dst_center[0] - EMOJI_WINDOW_SIZE_PX * anchor_center[0]
    top = dst_center[1] - EMOJI_WINDOW_SIZE_PX * anchor_center[1]
    return left, top, left + EMOJI_WINDOW_SIZE_PX, top + EMOJI_WINDOW_SIZE_PX


def global_source_bounds(rows, gt):
    xs = []
    ys = []
    for row in rows:
        xs.append(row["marker_window_x_px"])
        ys.append(row["marker_window_y_px"])
    for stimulus in EMOJI_ORDER:
        for feature in FEATURE_ORDER:
            x, y = gt[stimulus][feature]
            xs.append(x)
            ys.append(y)
        left, top, right, bottom = image_window_rect(stimulus, gt)
        xs.extend([left, right])
        ys.extend([top, bottom])
    return min(xs) - 35, max(xs) + 35, min(ys) - 35, max(ys) + 35


def transform_factory(source_bounds, box):
    min_x, max_x, min_y, max_y = source_bounds
    left, top, right, bottom = box
    scale = min((right - left) / (max_x - min_x), (bottom - top) / (max_y - min_y))
    used_w = (max_x - min_x) * scale
    used_h = (max_y - min_y) * scale
    offset_x = left + ((right - left) - used_w) / 2
    offset_y = top + ((bottom - top) - used_h) / 2

    def transform(x, y):
        return offset_x + (x - min_x) * scale, offset_y + (y - min_y) * scale

    return transform


def text_center(draw, x, y, text, fnt, fill=DARK):
    draw.text((x, y), text, font=fnt, fill=fill, anchor="mm")


def text_halo(draw, x, y, text, fnt, fill=DARK):
    halo = max(2, round(pt_to_px(0.35)))
    for dx, dy in [(-halo, 0), (halo, 0), (0, -halo), (0, halo)]:
        draw.text((x + dx, y + dy), text, font=fnt, fill=(255, 255, 255), anchor="mm")
    draw.text((x, y), text, font=fnt, fill=fill, anchor="mm")


def star_points(cx, cy, radius):
    points = []
    for i in range(10):
        angle = -math.pi / 2 + i * math.pi / 5
        r = radius if i % 2 == 0 else radius * 0.45
        points.append((cx + math.cos(angle) * r, cy + math.sin(angle) * r))
    return points


def draw_star_outline_pil(draw, cx, cy, radius, color=(32, 32, 32), width=None):
    width = width or max(2, round(pt_to_px(0.55)))
    points = star_points(cx, cy, radius)
    draw.line(points + [points[0]], fill=color, width=width, joint="curve")


def paste_emoji(canvas, stimulus, transform, gt, emoji_images, alpha_scale=0.34):
    img = emoji_images[stimulus].copy()
    left, top, right, bottom = image_window_rect(stimulus, gt)
    x0, y0 = transform(left, top)
    x1, y1 = transform(right, bottom)
    target_w = max(1, int(round(x1 - x0)))
    target_h = max(1, int(round(y1 - y0)))
    img = img.resize((target_w, target_h), Image.Resampling.LANCZOS)
    alpha = img.getchannel("A").point(lambda value: int(value * alpha_scale))
    img.putalpha(alpha)
    canvas.alpha_composite(img, (int(round(x0)), int(round(y0))))


def render_heatmap_background(stimulus, rows, gt, emoji_images, size, source_bounds):
    width, height = size
    panel = Image.new("RGBA", (round(width), round(height)), (255, 255, 255, 255))
    transform = transform_factory(source_bounds, (0, 0, width, height))
    paste_emoji(panel, stimulus, transform, gt, emoji_images)

    heat = Image.new("RGBA", panel.size, (255, 255, 255, 0))
    heat_draw = ImageDraw.Draw(heat)
    radius = width * (22.0 / 360.0)
    for row in rows:
        if row["stimulus"] != stimulus:
            continue
        x, y = transform(row["marker_window_x_px"], row["marker_window_y_px"])
        heat_draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=(220, 30, 50, 68))
    heat = heat.filter(ImageFilter.GaussianBlur(max(1, round(width * (9.0 / 360.0)))))
    panel.alpha_composite(heat)
    return panel


def image_data_uri(img):
    buffer = io.BytesIO()
    img.convert("RGB").save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def save_bitmap(canvas, stem):
    rgb = canvas.convert("RGB")
    png = stem.with_suffix(".png")
    tiff = stem.with_suffix(".tiff")
    rgb.save(png, quality=95, dpi=(EXPORT_DPI, EXPORT_DPI))
    rgb.save(tiff, compression="tiff_lzw", dpi=(EXPORT_DPI, EXPORT_DPI))
    return png, tiff


def layout_fig5c(width, height):
    margin_x = mm_to_px(15.0)
    gap = mm_to_px(10.0)
    panel_w = (width - 2 * margin_x - 2 * gap) / 3
    heat_top = mm_to_px(26.0)
    heat_bottom = mm_to_px(56.0)
    box_top = mm_to_px(64.0)
    box_bottom = mm_to_px(99.0)
    panels = []
    for idx, stimulus in enumerate(EMOJI_ORDER):
        left = margin_x + idx * (panel_w + gap)
        panels.append(
            {
                "stimulus": stimulus,
                "left": left,
                "right": left + panel_w,
                "heat_box": (left, heat_top, left + panel_w, heat_bottom),
                "box_box": (left, box_top, left + panel_w, box_bottom),
                "title_y": mm_to_px(22.0),
            }
        )
    return panels


def layout_heatmap(width, height):
    margin_x = mm_to_px(15.0)
    gap = mm_to_px(10.0)
    panel_w = (width - 2 * margin_x - 2 * gap) / 3
    heat_top = mm_to_px(24.0)
    heat_bottom = mm_to_px(51.0)
    panels = []
    for idx, stimulus in enumerate(EMOJI_ORDER):
        left = margin_x + idx * (panel_w + gap)
        panels.append(
            {
                "stimulus": stimulus,
                "left": left,
                "right": left + panel_w,
                "heat_box": (left, heat_top, left + panel_w, heat_bottom),
                "title_y": mm_to_px(20.0),
            }
        )
    return panels


def draw_marker_legend_pil(draw, center_x, y):
    parts = [
        ("red", "red density = participant marker density"),
        ("dot", "blue dots = participants"),
        ("star", "black star = target point"),
    ]
    font = FONTS["legend"]
    widths = []
    marker_gap = mm_to_px(2.0)
    part_gap = mm_to_px(6.0)
    for marker, text in parts:
        bbox = draw.textbbox((0, 0), text, font=font)
        widths.append(mm_to_px(4.0) + marker_gap + (bbox[2] - bbox[0]))
    total = sum(widths) + part_gap * (len(parts) - 1)
    x = center_x - total / 2
    for (marker, text), width in zip(parts, widths):
        mx = x + mm_to_px(1.5)
        if marker == "red":
            draw.ellipse((mx - mm_to_px(1.1), y - mm_to_px(1.1), mx + mm_to_px(1.1), y + mm_to_px(1.1)), fill=(220, 30, 50, 110))
        elif marker == "dot":
            r = mm_to_px(0.75)
            draw.ellipse((mx - r, y - r, mx + r, y + r), fill=(10, 92, 180), outline=(255, 255, 255), width=max(1, round(pt_to_px(0.35))))
        else:
            draw_star_outline_pil(draw, mx, y, mm_to_px(1.25), color=(32, 32, 32), width=max(1, round(pt_to_px(0.45))))
        draw.text((x + mm_to_px(4.0) + marker_gap, y), text, font=font, fill=(40, 40, 40), anchor="lm")
        x += width + part_gap


def draw_heatmap_overlay_pil(draw, stimulus, rows, gt, box, source_bounds):
    left, top, right, bottom = box
    transform = transform_factory(source_bounds, (left, top, right, bottom))
    dot_r = mm_to_px(0.45)
    dot_outline = max(1, round(pt_to_px(0.32)))
    for row in rows:
        if row["stimulus"] != stimulus:
            continue
        x, y = transform(row["marker_window_x_px"], row["marker_window_y_px"])
        draw.ellipse((x - dot_r, y - dot_r, x + dot_r, y + dot_r), fill=(10, 92, 180), outline=(255, 255, 255), width=dot_outline)

    offsets = {"L": (-mm_to_px(1.4), mm_to_px(1.9)), "R": (mm_to_px(1.4), mm_to_px(1.9)), "M": (0, mm_to_px(2.15))}
    for feature in FEATURE_ORDER:
        tx, ty = gt[stimulus][feature]
        x, y = transform(tx, ty)
        draw_star_outline_pil(draw, x, y, mm_to_px(1.55), color=(32, 32, 32))
        ox, oy = offsets[feature]
        text_halo(draw, x + ox, y + oy, feature, FONTS["feature"], fill=(25, 25, 25))


def grouped_distances(detail, stimulus, feature):
    return [
        row["distance_percent"]
        for row in detail
        if row["stimulus"] == stimulus and row["feature"] == feature
    ]


def draw_boxplot_panel_pil(draw, stimulus, detail, box):
    left, top, right, bottom = box
    plot_left = left + mm_to_px(7.0)
    plot_right = right - mm_to_px(3.5)
    plot_top = top + mm_to_px(3.0)
    plot_bottom = bottom - mm_to_px(7.0)
    y_max = ERROR_Y_MAX_PERCENT
    axis_w = max(1, round(pt_to_px(0.6)))
    grid_w = max(1, round(pt_to_px(0.3)))

    def y_pos(value):
        return plot_bottom - (value / y_max) * (plot_bottom - plot_top)

    for tick in ERROR_Y_TICKS_PERCENT:
        y = y_pos(tick)
        if tick > 0:
            draw.line((plot_left, y, plot_right, y), fill=(220, 220, 220), width=grid_w)
        draw.line((plot_left - mm_to_px(1.0), y, plot_left, y), fill=(85, 85, 85), width=axis_w)
        draw.text((plot_left - mm_to_px(2.2), y), str(tick), font=FONTS["tick"], fill=(40, 40, 40), anchor="rm")

    draw.line((plot_left, plot_top, plot_left, plot_bottom), fill=(85, 85, 85), width=axis_w)
    draw.line((plot_left, plot_bottom, plot_right, plot_bottom), fill=(85, 85, 85), width=axis_w)

    box_w = mm_to_px(5.0)
    cap_w = mm_to_px(3.0)
    dot_r = mm_to_px(0.45)
    jitter = [-mm_to_px(1.1), mm_to_px(0.8), -mm_to_px(0.35), mm_to_px(1.3), -mm_to_px(1.55), mm_to_px(0.25), mm_to_px(1.7)]
    for idx, feature in enumerate(FEATURE_ORDER):
        x = plot_left + (idx + 0.5) * (plot_right - plot_left) / 3
        values = grouped_distances(detail, stimulus, feature)
        if values:
            stats = box_stats(values)
            draw.line((x, y_pos(stats["min"]), x, y_pos(stats["max"])), fill=(85, 85, 85), width=axis_w)
            draw.line((x - cap_w / 2, y_pos(stats["min"]), x + cap_w / 2, y_pos(stats["min"])), fill=(85, 85, 85), width=axis_w)
            draw.line((x - cap_w / 2, y_pos(stats["max"]), x + cap_w / 2, y_pos(stats["max"])), fill=(85, 85, 85), width=axis_w)
            draw.rectangle((x - box_w / 2, y_pos(stats["q3"]), x + box_w / 2, y_pos(stats["q1"])), fill=(183, 213, 240), outline=(55, 55, 55), width=axis_w)
            draw.line((x - box_w / 2, y_pos(stats["median"]), x + box_w / 2, y_pos(stats["median"])), fill=(20, 20, 20), width=max(axis_w, round(pt_to_px(0.75))))
            for i, value in enumerate(values):
                px = x + jitter[i % len(jitter)]
                py = y_pos(value)
                draw.ellipse((px - dot_r, py - dot_r, px + dot_r, py + dot_r), fill=(10, 92, 180), outline=(255, 255, 255), width=max(1, round(pt_to_px(0.3))))
        text_center(draw, x, plot_bottom + mm_to_px(4.0), feature, FONTS["tick"], fill=(35, 35, 35))


def draw_rotated_y_label_pil(canvas, x, y, text):
    label_font = FONTS["axis"]
    bbox = ImageDraw.Draw(Image.new("RGBA", (1, 1))).textbbox((0, 0), text, font=label_font)
    layer = Image.new("RGBA", (bbox[2] - bbox[0] + 20, bbox[3] - bbox[1] + 20), (255, 255, 255, 0))
    layer_draw = ImageDraw.Draw(layer)
    layer_draw.text((10, 10), text, font=label_font, fill=(30, 30, 30))
    layer = layer.rotate(90, expand=True)
    canvas.alpha_composite(layer, (round(x - layer.width / 2), round(y - layer.height / 2)))


def draw_fig5c_png(rows, detail, gt, emoji_images):
    width = round(mm_to_px(FIG_WIDTH_MM))
    height = round(mm_to_px(FIG5C_HEIGHT_MM))
    canvas = Image.new("RGBA", (width, height), (255, 255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    panels = layout_fig5c(width, height)
    source_bounds = global_source_bounds(rows, gt)

    draw.text((mm_to_px(3.0), mm_to_px(3.0)), "C", font=FONTS["panel"], fill=(10, 10, 10), anchor="la")
    text_center(draw, width / 2, mm_to_px(6.5), "Localization without visual access", FONTS["title"], fill=(20, 20, 20))
    n = len(set(row["participant_id"] for row in rows))
    text_center(draw, width / 2, mm_to_px(11.5), f"Blind participants under IHF condition; n = {n}", FONTS["subtitle"], fill=(35, 35, 35))
    draw_marker_legend_pil(draw, width / 2, mm_to_px(16.0))

    for panel in panels:
        stimulus = panel["stimulus"]
        heat_box = panel["heat_box"]
        text_center(draw, (heat_box[0] + heat_box[2]) / 2, panel["title_y"], stimulus.title(), FONTS["emoji_title"], fill=(20, 20, 20))
        bg = render_heatmap_background(
            stimulus,
            rows,
            gt,
            emoji_images,
            (heat_box[2] - heat_box[0], heat_box[3] - heat_box[1]),
            source_bounds,
        )
        canvas.alpha_composite(bg, (round(heat_box[0]), round(heat_box[1])))
        draw_heatmap_overlay_pil(draw, stimulus, rows, gt, heat_box, source_bounds)
        draw_boxplot_panel_pil(draw, stimulus, detail, panel["box_box"])

    first_box = panels[0]["box_box"]
    draw_rotated_y_label_pil(
        canvas,
        mm_to_px(6.6),
        (first_box[1] + first_box[3]) / 2,
        "Localization error (% of window diagonal)",
    )
    note = "L = left eye, R = right eye, M = mouth; window = 665 x 440 px (diagonal = 797.386 px)."
    text_center(draw, width / 2, mm_to_px(108.5), note, FONTS["note"], fill=(35, 35, 35))
    return canvas


def draw_heatmap_png(rows, gt, emoji_images):
    width = round(mm_to_px(FIG_WIDTH_MM))
    height = round(mm_to_px(HEATMAP_HEIGHT_MM))
    canvas = Image.new("RGBA", (width, height), (255, 255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    panels = layout_heatmap(width, height)
    source_bounds = global_source_bounds(rows, gt)

    text_center(draw, width / 2, mm_to_px(5.3), "Blind users: point distribution heatmap by emoji", FONTS["title"], fill=(20, 20, 20))
    n = len(set(row["participant_id"] for row in rows))
    text_center(draw, width / 2, mm_to_px(9.7), f"Blind participants under IHF condition; n = {n}", FONTS["subtitle"], fill=(35, 35, 35))
    draw_marker_legend_pil(draw, width / 2, mm_to_px(14.0))

    for panel in panels:
        stimulus = panel["stimulus"]
        heat_box = panel["heat_box"]
        text_center(draw, (heat_box[0] + heat_box[2]) / 2, panel["title_y"], stimulus.title(), FONTS["emoji_title"], fill=(20, 20, 20))
        bg = render_heatmap_background(
            stimulus,
            rows,
            gt,
            emoji_images,
            (heat_box[2] - heat_box[0], heat_box[3] - heat_box[1]),
            source_bounds,
        )
        canvas.alpha_composite(bg, (round(heat_box[0]), round(heat_box[1])))
        draw_heatmap_overlay_pil(draw, stimulus, rows, gt, heat_box, source_bounds)

    note = "L = left eye, R = right eye, M = mouth; lower values indicate better localization."
    text_center(draw, width / 2, mm_to_px(58.0), note, FONTS["note"], fill=(35, 35, 35))
    return canvas


def svg_text(x, y, text, style, anchor="middle", extra=""):
    size_pt, bold = FONT_PT[style]
    weight = "700" if bold else "400"
    return (
        f'<text x="{x:.2f}" y="{y:.2f}" text-anchor="{anchor}" '
        f'dominant-baseline="middle" font-family="Arial, Helvetica, sans-serif" '
        f'font-size="{pt_to_px(size_pt):.2f}px" font-weight="{weight}" fill="{DARK}" {extra}>'
        f"{escape(text)}</text>"
    )


def svg_text_halo(x, y, text, style, anchor="middle"):
    size_pt, bold = FONT_PT[style]
    weight = "700" if bold else "400"
    common = (
        f'x="{x:.2f}" y="{y:.2f}" text-anchor="{anchor}" dominant-baseline="middle" '
        f'font-family="Arial, Helvetica, sans-serif" font-size="{pt_to_px(size_pt):.2f}px" '
        f'font-weight="{weight}"'
    )
    escaped = escape(text)
    return (
        f'<text {common} fill="{DARK}" stroke="#FFFFFF" stroke-width="{pt_to_px(0.75):.2f}" '
        f'stroke-linejoin="round" paint-order="stroke fill">{escaped}</text>'
    )


def svg_star(cx, cy, radius):
    points = " ".join(f"{x:.2f},{y:.2f}" for x, y in star_points(cx, cy, radius))
    return f'<polygon points="{points}" fill="none" stroke="{DARK}" stroke-width="{pt_to_px(0.55):.2f}" stroke-linejoin="round"/>'


def svg_marker_legend(center_x, y):
    parts = [
        ("red", "red density = participant marker density"),
        ("dot", "blue dots = participants"),
        ("star", "black star = target point"),
    ]
    avg_char_w = pt_to_px(7.8) * 0.52
    marker_gap = mm_to_px(2.0)
    part_gap = mm_to_px(6.0)
    widths = [mm_to_px(4.0) + marker_gap + len(text) * avg_char_w for _, text in parts]
    total = sum(widths) + part_gap * (len(parts) - 1)
    x = center_x - total / 2
    out = []
    for (marker, text), width in zip(parts, widths):
        mx = x + mm_to_px(1.5)
        if marker == "red":
            out.append(f'<circle cx="{mx:.2f}" cy="{y:.2f}" r="{mm_to_px(1.1):.2f}" fill="{RED}" fill-opacity="0.45"/>')
        elif marker == "dot":
            out.append(f'<circle cx="{mx:.2f}" cy="{y:.2f}" r="{mm_to_px(0.75):.2f}" fill="{BLUE}" stroke="#FFFFFF" stroke-width="{pt_to_px(0.35):.2f}"/>')
        else:
            out.append(svg_star(mx, y, mm_to_px(1.25)))
        out.append(svg_text(x + mm_to_px(4.0) + marker_gap, y, text, "legend", anchor="start"))
        x += width + part_gap
    return "\n".join(out)


def svg_heatmap_overlay(stimulus, rows, gt, box, source_bounds):
    left, top, right, bottom = box
    transform = transform_factory(source_bounds, (left, top, right, bottom))
    out = []
    for row in rows:
        if row["stimulus"] != stimulus:
            continue
        x, y = transform(row["marker_window_x_px"], row["marker_window_y_px"])
        out.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="{mm_to_px(0.45):.2f}" fill="{BLUE}" stroke="#FFFFFF" stroke-width="{pt_to_px(0.32):.2f}"/>')
    offsets = {"L": (-mm_to_px(1.4), mm_to_px(1.9)), "R": (mm_to_px(1.4), mm_to_px(1.9)), "M": (0, mm_to_px(2.15))}
    for feature in FEATURE_ORDER:
        x, y = transform(*gt[stimulus][feature])
        out.append(svg_star(x, y, mm_to_px(1.55)))
        ox, oy = offsets[feature]
        out.append(svg_text_halo(x + ox, y + oy, feature, "feature"))
    return "\n".join(out)


def svg_boxplot_panel(stimulus, detail, box):
    left, top, right, bottom = box
    plot_left = left + mm_to_px(7.0)
    plot_right = right - mm_to_px(3.5)
    plot_top = top + mm_to_px(3.0)
    plot_bottom = bottom - mm_to_px(7.0)
    y_max = ERROR_Y_MAX_PERCENT
    axis_w = pt_to_px(0.6)
    grid_w = pt_to_px(0.3)
    out = []

    def y_pos(value):
        return plot_bottom - (value / y_max) * (plot_bottom - plot_top)

    for tick in ERROR_Y_TICKS_PERCENT:
        y = y_pos(tick)
        if tick > 0:
            out.append(f'<line x1="{plot_left:.2f}" y1="{y:.2f}" x2="{plot_right:.2f}" y2="{y:.2f}" stroke="{GRID}" stroke-width="{grid_w:.2f}"/>')
        out.append(f'<line x1="{plot_left - mm_to_px(1.0):.2f}" y1="{y:.2f}" x2="{plot_left:.2f}" y2="{y:.2f}" stroke="{AXIS}" stroke-width="{axis_w:.2f}"/>')
        out.append(svg_text(plot_left - mm_to_px(2.2), y, str(tick), "tick", anchor="end"))
    out.append(f'<line x1="{plot_left:.2f}" y1="{plot_top:.2f}" x2="{plot_left:.2f}" y2="{plot_bottom:.2f}" stroke="{AXIS}" stroke-width="{axis_w:.2f}"/>')
    out.append(f'<line x1="{plot_left:.2f}" y1="{plot_bottom:.2f}" x2="{plot_right:.2f}" y2="{plot_bottom:.2f}" stroke="{AXIS}" stroke-width="{axis_w:.2f}"/>')

    box_w = mm_to_px(5.0)
    cap_w = mm_to_px(3.0)
    jitter = [-mm_to_px(1.1), mm_to_px(0.8), -mm_to_px(0.35), mm_to_px(1.3), -mm_to_px(1.55), mm_to_px(0.25), mm_to_px(1.7)]
    for idx, feature in enumerate(FEATURE_ORDER):
        x = plot_left + (idx + 0.5) * (plot_right - plot_left) / 3
        values = grouped_distances(detail, stimulus, feature)
        if values:
            stats = box_stats(values)
            out.append(f'<line x1="{x:.2f}" y1="{y_pos(stats["min"]):.2f}" x2="{x:.2f}" y2="{y_pos(stats["max"]):.2f}" stroke="{AXIS}" stroke-width="{axis_w:.2f}"/>')
            out.append(f'<line x1="{x - cap_w / 2:.2f}" y1="{y_pos(stats["min"]):.2f}" x2="{x + cap_w / 2:.2f}" y2="{y_pos(stats["min"]):.2f}" stroke="{AXIS}" stroke-width="{axis_w:.2f}"/>')
            out.append(f'<line x1="{x - cap_w / 2:.2f}" y1="{y_pos(stats["max"]):.2f}" x2="{x + cap_w / 2:.2f}" y2="{y_pos(stats["max"]):.2f}" stroke="{AXIS}" stroke-width="{axis_w:.2f}"/>')
            out.append(f'<rect x="{x - box_w / 2:.2f}" y="{y_pos(stats["q3"]):.2f}" width="{box_w:.2f}" height="{y_pos(stats["q1"]) - y_pos(stats["q3"]):.2f}" fill="{BOX_FILL}" stroke="#373737" stroke-width="{axis_w:.2f}"/>')
            out.append(f'<line x1="{x - box_w / 2:.2f}" y1="{y_pos(stats["median"]):.2f}" x2="{x + box_w / 2:.2f}" y2="{y_pos(stats["median"]):.2f}" stroke="#141414" stroke-width="{pt_to_px(0.75):.2f}"/>')
            for i, value in enumerate(values):
                px = x + jitter[i % len(jitter)]
                py = y_pos(value)
                out.append(f'<circle cx="{px:.2f}" cy="{py:.2f}" r="{mm_to_px(0.45):.2f}" fill="{BLUE}" stroke="#FFFFFF" stroke-width="{pt_to_px(0.3):.2f}"/>')
        out.append(svg_text(x, plot_bottom + mm_to_px(4.0), feature, "tick"))
    return "\n".join(out)


def write_fig5c_svg(rows, detail, gt, emoji_images, stem):
    width = round(mm_to_px(FIG_WIDTH_MM))
    height = round(mm_to_px(FIG5C_HEIGHT_MM))
    panels = layout_fig5c(width, height)
    source_bounds = global_source_bounds(rows, gt)
    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{FIG_WIDTH_MM}mm" height="{FIG5C_HEIGHT_MM}mm" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#FFFFFF"/>',
        svg_text(mm_to_px(3.0), mm_to_px(5.5), "C", "panel", anchor="start"),
        svg_text(width / 2, mm_to_px(6.5), "Localization without visual access", "title"),
    ]
    n = len(set(row["participant_id"] for row in rows))
    out.append(svg_text(width / 2, mm_to_px(11.5), f"Blind participants under IHF condition; n = {n}", "subtitle"))
    out.append(svg_marker_legend(width / 2, mm_to_px(16.0)))
    for panel in panels:
        stimulus = panel["stimulus"]
        heat_box = panel["heat_box"]
        out.append(svg_text((heat_box[0] + heat_box[2]) / 2, panel["title_y"], stimulus.title(), "emoji_title"))
        bg = render_heatmap_background(
            stimulus,
            rows,
            gt,
            emoji_images,
            (heat_box[2] - heat_box[0], heat_box[3] - heat_box[1]),
            source_bounds,
        )
        out.append(f'<image x="{heat_box[0]:.2f}" y="{heat_box[1]:.2f}" width="{heat_box[2] - heat_box[0]:.2f}" height="{heat_box[3] - heat_box[1]:.2f}" href="{image_data_uri(bg)}"/>')
        out.append(svg_heatmap_overlay(stimulus, rows, gt, heat_box, source_bounds))
        out.append(svg_boxplot_panel(stimulus, detail, panel["box_box"]))
    first_box = panels[0]["box_box"]
    y_label_x = mm_to_px(6.6)
    y_label_y = (first_box[1] + first_box[3]) / 2
    out.append(
        f'<g transform="translate({y_label_x:.2f},{y_label_y:.2f}) rotate(-90)">'
        + svg_text(0, 0, "Localization error (% of window diagonal)", "axis")
        + "</g>"
    )
    out.append(svg_text(width / 2, mm_to_px(108.5), "L = left eye, R = right eye, M = mouth; lower values indicate better localization.", "note"))
    out.append("</svg>")
    path = stem.with_suffix(".svg")
    path.write_text("\n".join(out), encoding="utf-8")
    return path


def write_heatmap_svg(rows, gt, emoji_images, stem):
    width = round(mm_to_px(FIG_WIDTH_MM))
    height = round(mm_to_px(HEATMAP_HEIGHT_MM))
    panels = layout_heatmap(width, height)
    source_bounds = global_source_bounds(rows, gt)
    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{FIG_WIDTH_MM}mm" height="{HEATMAP_HEIGHT_MM}mm" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#FFFFFF"/>',
        svg_text(width / 2, mm_to_px(5.3), "Blind users: point distribution heatmap by emoji", "title"),
    ]
    n = len(set(row["participant_id"] for row in rows))
    out.append(svg_text(width / 2, mm_to_px(9.7), f"Blind participants under IHF condition; n = {n}", "subtitle"))
    out.append(svg_marker_legend(width / 2, mm_to_px(14.0)))
    for panel in panels:
        stimulus = panel["stimulus"]
        heat_box = panel["heat_box"]
        out.append(svg_text((heat_box[0] + heat_box[2]) / 2, panel["title_y"], stimulus.title(), "emoji_title"))
        bg = render_heatmap_background(
            stimulus,
            rows,
            gt,
            emoji_images,
            (heat_box[2] - heat_box[0], heat_box[3] - heat_box[1]),
            source_bounds,
        )
        out.append(f'<image x="{heat_box[0]:.2f}" y="{heat_box[1]:.2f}" width="{heat_box[2] - heat_box[0]:.2f}" height="{heat_box[3] - heat_box[1]:.2f}" href="{image_data_uri(bg)}"/>')
        out.append(svg_heatmap_overlay(stimulus, rows, gt, heat_box, source_bounds))
    out.append(svg_text(width / 2, mm_to_px(58.0), "L = left eye, R = right eye, M = mouth; lower values indicate better localization.", "note"))
    out.append("</svg>")
    path = stem.with_suffix(".svg")
    path.write_text("\n".join(out), encoding="utf-8")
    return path


def pdf_y(y, page_h_pt):
    return page_h_pt - px_to_pt(y)


def pdf_set_font(c, style):
    size_pt, bold = FONT_PT[style]
    c.setFont("Helvetica-Bold" if bold else "Helvetica", size_pt)
    return size_pt


def pdf_text_center(c, x, y, text, style):
    size_pt = pdf_set_font(c, style)
    c.setFillColorRGB(32 / 255, 32 / 255, 32 / 255)
    c.drawCentredString(px_to_pt(x), pdf_y(y, c._pagesize[1]) - size_pt * 0.35, text)


def pdf_text_left(c, x, y, text, style):
    size_pt = pdf_set_font(c, style)
    c.setFillColorRGB(32 / 255, 32 / 255, 32 / 255)
    c.drawString(px_to_pt(x), pdf_y(y, c._pagesize[1]) - size_pt * 0.35, text)


def pdf_text_halo(c, x, y, text, style):
    size_pt = pdf_set_font(c, style)
    page_h = c._pagesize[1]
    offsets = [(-0.35, 0), (0.35, 0), (0, -0.35), (0, 0.35)]
    c.setFillColorRGB(1, 1, 1)
    for ox, oy in offsets:
        c.drawCentredString(px_to_pt(x) + ox, pdf_y(y, page_h) - size_pt * 0.35 + oy, text)
    c.setFillColorRGB(32 / 255, 32 / 255, 32 / 255)
    c.drawCentredString(px_to_pt(x), pdf_y(y, page_h) - size_pt * 0.35, text)


def pdf_circle(c, x, y, r, fill_hex, stroke_hex=None, stroke_w=0):
    def hex_to_rgb(value):
        value = value.lstrip("#")
        return tuple(int(value[i : i + 2], 16) / 255 for i in (0, 2, 4))

    c.setFillColorRGB(*hex_to_rgb(fill_hex))
    if stroke_hex and stroke_w:
        c.setStrokeColorRGB(*hex_to_rgb(stroke_hex))
        c.setLineWidth(px_to_pt(stroke_w))
        stroke = 1
    else:
        stroke = 0
    c.circle(px_to_pt(x), pdf_y(y, c._pagesize[1]), px_to_pt(r), fill=1, stroke=stroke)


def pdf_line(c, x1, y1, x2, y2, color_hex, width_px):
    value = color_hex.lstrip("#")
    rgb = tuple(int(value[i : i + 2], 16) / 255 for i in (0, 2, 4))
    c.setStrokeColorRGB(*rgb)
    c.setLineWidth(px_to_pt(width_px))
    c.line(px_to_pt(x1), pdf_y(y1, c._pagesize[1]), px_to_pt(x2), pdf_y(y2, c._pagesize[1]))


def pdf_star(c, cx, cy, radius):
    pts = star_points(cx, cy, radius)
    path = c.beginPath()
    path.moveTo(px_to_pt(pts[0][0]), pdf_y(pts[0][1], c._pagesize[1]))
    for x, y in pts[1:]:
        path.lineTo(px_to_pt(x), pdf_y(y, c._pagesize[1]))
    path.close()
    c.setLineWidth(px_to_pt(pt_to_px(0.55)))
    c.setStrokeColorRGB(32 / 255, 32 / 255, 32 / 255)
    c.drawPath(path, stroke=1, fill=0)


def pdf_draw_image(c, img, box):
    left, top, right, bottom = box
    buffer = io.BytesIO()
    img.convert("RGB").save(buffer, format="PNG")
    buffer.seek(0)
    c.drawImage(
        ImageReader(buffer),
        px_to_pt(left),
        pdf_y(bottom, c._pagesize[1]),
        width=px_to_pt(right - left),
        height=px_to_pt(bottom - top),
        mask="auto",
    )


def pdf_marker_legend(c, center_x, y):
    parts = [
        ("red", "red density = participant marker density"),
        ("dot", "blue dots = participants"),
        ("star", "black star = target point"),
    ]
    avg_char_w = pt_to_px(7.8) * 0.52
    marker_gap = mm_to_px(2.0)
    part_gap = mm_to_px(6.0)
    widths = [mm_to_px(4.0) + marker_gap + len(text) * avg_char_w for _, text in parts]
    total = sum(widths) + part_gap * (len(parts) - 1)
    x = center_x - total / 2
    for (marker, text), width in zip(parts, widths):
        mx = x + mm_to_px(1.5)
        if marker == "red":
            pdf_circle(c, mx, y, mm_to_px(1.1), RED)
        elif marker == "dot":
            pdf_circle(c, mx, y, mm_to_px(0.75), BLUE, "#FFFFFF", pt_to_px(0.35))
        else:
            pdf_star(c, mx, y, mm_to_px(1.25))
        pdf_text_left(c, x + mm_to_px(4.0) + marker_gap, y, text, "legend")
        x += width + part_gap


def pdf_heatmap_overlay(c, stimulus, rows, gt, box, source_bounds):
    left, top, right, bottom = box
    transform = transform_factory(source_bounds, (left, top, right, bottom))
    for row in rows:
        if row["stimulus"] != stimulus:
            continue
        x, y = transform(row["marker_window_x_px"], row["marker_window_y_px"])
        pdf_circle(c, x, y, mm_to_px(0.45), BLUE, "#FFFFFF", pt_to_px(0.32))
    offsets = {"L": (-mm_to_px(1.4), mm_to_px(1.9)), "R": (mm_to_px(1.4), mm_to_px(1.9)), "M": (0, mm_to_px(2.15))}
    for feature in FEATURE_ORDER:
        x, y = transform(*gt[stimulus][feature])
        pdf_star(c, x, y, mm_to_px(1.55))
        ox, oy = offsets[feature]
        pdf_text_halo(c, x + ox, y + oy, feature, "feature")


def pdf_boxplot_panel(c, stimulus, detail, box):
    left, top, right, bottom = box
    plot_left = left + mm_to_px(7.0)
    plot_right = right - mm_to_px(3.5)
    plot_top = top + mm_to_px(3.0)
    plot_bottom = bottom - mm_to_px(7.0)
    y_max = ERROR_Y_MAX_PERCENT
    axis_w = pt_to_px(0.6)
    grid_w = pt_to_px(0.3)

    def y_pos(value):
        return plot_bottom - (value / y_max) * (plot_bottom - plot_top)

    for tick in ERROR_Y_TICKS_PERCENT:
        y = y_pos(tick)
        if tick > 0:
            pdf_line(c, plot_left, y, plot_right, y, GRID, grid_w)
        pdf_line(c, plot_left - mm_to_px(1.0), y, plot_left, y, AXIS, axis_w)
        size_pt = pdf_set_font(c, "tick")
        c.setFillColorRGB(32 / 255, 32 / 255, 32 / 255)
        c.drawRightString(px_to_pt(plot_left - mm_to_px(2.2)), pdf_y(y, c._pagesize[1]) - size_pt * 0.35, str(tick))
    pdf_line(c, plot_left, plot_top, plot_left, plot_bottom, AXIS, axis_w)
    pdf_line(c, plot_left, plot_bottom, plot_right, plot_bottom, AXIS, axis_w)

    box_w = mm_to_px(5.0)
    cap_w = mm_to_px(3.0)
    jitter = [-mm_to_px(1.1), mm_to_px(0.8), -mm_to_px(0.35), mm_to_px(1.3), -mm_to_px(1.55), mm_to_px(0.25), mm_to_px(1.7)]
    for idx, feature in enumerate(FEATURE_ORDER):
        x = plot_left + (idx + 0.5) * (plot_right - plot_left) / 3
        values = grouped_distances(detail, stimulus, feature)
        if values:
            stats = box_stats(values)
            pdf_line(c, x, y_pos(stats["min"]), x, y_pos(stats["max"]), AXIS, axis_w)
            pdf_line(c, x - cap_w / 2, y_pos(stats["min"]), x + cap_w / 2, y_pos(stats["min"]), AXIS, axis_w)
            pdf_line(c, x - cap_w / 2, y_pos(stats["max"]), x + cap_w / 2, y_pos(stats["max"]), AXIS, axis_w)
            c.setFillColorRGB(183 / 255, 213 / 255, 240 / 255)
            c.setStrokeColorRGB(55 / 255, 55 / 255, 55 / 255)
            c.setLineWidth(px_to_pt(axis_w))
            rect_y = pdf_y(y_pos(stats["q1"]), c._pagesize[1])
            rect_h = px_to_pt(y_pos(stats["q1"]) - y_pos(stats["q3"]))
            c.rect(px_to_pt(x - box_w / 2), rect_y, px_to_pt(box_w), rect_h, fill=1, stroke=1)
            pdf_line(c, x - box_w / 2, y_pos(stats["median"]), x + box_w / 2, y_pos(stats["median"]), "#141414", pt_to_px(0.75))
            for i, value in enumerate(values):
                px = x + jitter[i % len(jitter)]
                py = y_pos(value)
                pdf_circle(c, px, py, mm_to_px(0.45), BLUE, "#FFFFFF", pt_to_px(0.3))
        pdf_text_center(c, x, plot_bottom + mm_to_px(4.0), feature, "tick")


def write_fig5c_pdf(rows, detail, gt, emoji_images, stem):
    page_w_pt = FIG_WIDTH_MM / 25.4 * 72
    page_h_pt = FIG5C_HEIGHT_MM / 25.4 * 72
    width = round(mm_to_px(FIG_WIDTH_MM))
    height = round(mm_to_px(FIG5C_HEIGHT_MM))
    path = stem.with_suffix(".pdf")
    c = pdf_canvas.Canvas(str(path), pagesize=(page_w_pt, page_h_pt))
    panels = layout_fig5c(width, height)
    source_bounds = global_source_bounds(rows, gt)
    pdf_text_left(c, mm_to_px(3.0), mm_to_px(5.5), "C", "panel")
    pdf_text_center(c, width / 2, mm_to_px(6.5), "Localization without visual access", "title")
    n = len(set(row["participant_id"] for row in rows))
    pdf_text_center(c, width / 2, mm_to_px(11.5), f"Blind participants under IHF condition; n = {n}", "subtitle")
    pdf_marker_legend(c, width / 2, mm_to_px(16.0))
    for panel in panels:
        stimulus = panel["stimulus"]
        heat_box = panel["heat_box"]
        pdf_text_center(c, (heat_box[0] + heat_box[2]) / 2, panel["title_y"], stimulus.title(), "emoji_title")
        bg = render_heatmap_background(
            stimulus,
            rows,
            gt,
            emoji_images,
            (heat_box[2] - heat_box[0], heat_box[3] - heat_box[1]),
            source_bounds,
        )
        pdf_draw_image(c, bg, heat_box)
        pdf_heatmap_overlay(c, stimulus, rows, gt, heat_box, source_bounds)
        pdf_boxplot_panel(c, stimulus, detail, panel["box_box"])
    first_box = panels[0]["box_box"]
    c.saveState()
    c.translate(px_to_pt(mm_to_px(6.6)), pdf_y((first_box[1] + first_box[3]) / 2, page_h_pt))
    c.rotate(90)
    pdf_set_font(c, "axis")
    c.setFillColorRGB(32 / 255, 32 / 255, 32 / 255)
    c.drawCentredString(0, -FONT_PT["axis"][0] * 0.35, "Localization error (% of window diagonal)")
    c.restoreState()
    pdf_text_center(c, width / 2, mm_to_px(108.5), "L = left eye, R = right eye, M = mouth; lower values indicate better localization.", "note")
    c.showPage()
    c.save()
    return path


def write_heatmap_pdf(rows, gt, emoji_images, stem):
    page_w_pt = FIG_WIDTH_MM / 25.4 * 72
    page_h_pt = HEATMAP_HEIGHT_MM / 25.4 * 72
    width = round(mm_to_px(FIG_WIDTH_MM))
    height = round(mm_to_px(HEATMAP_HEIGHT_MM))
    path = stem.with_suffix(".pdf")
    c = pdf_canvas.Canvas(str(path), pagesize=(page_w_pt, page_h_pt))
    panels = layout_heatmap(width, height)
    source_bounds = global_source_bounds(rows, gt)
    pdf_text_center(c, width / 2, mm_to_px(5.3), "Blind users: point distribution heatmap by emoji", "title")
    n = len(set(row["participant_id"] for row in rows))
    pdf_text_center(c, width / 2, mm_to_px(9.7), f"Blind participants under IHF condition; n = {n}", "subtitle")
    pdf_marker_legend(c, width / 2, mm_to_px(14.0))
    for panel in panels:
        stimulus = panel["stimulus"]
        heat_box = panel["heat_box"]
        pdf_text_center(c, (heat_box[0] + heat_box[2]) / 2, panel["title_y"], stimulus.title(), "emoji_title")
        bg = render_heatmap_background(
            stimulus,
            rows,
            gt,
            emoji_images,
            (heat_box[2] - heat_box[0], heat_box[3] - heat_box[1]),
            source_bounds,
        )
        pdf_draw_image(c, bg, heat_box)
        pdf_heatmap_overlay(c, stimulus, rows, gt, heat_box, source_bounds)
    pdf_text_center(c, width / 2, mm_to_px(58.0), "L = left eye, R = right eye, M = mouth; lower values indicate better localization.", "note")
    c.showPage()
    c.save()
    return path


def validate_counts(rows):
    participants = sorted(set(row["participant_id"] for row in rows))
    counts = {}
    for stimulus in EMOJI_ORDER:
        for feature in FEATURE_ORDER:
            counts[(stimulus, feature)] = len(
                {
                    row["participant_id"]
                    for row in rows
                    if row["stimulus"] == stimulus and row["feature"] == feature
                }
            )
    return participants, counts


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    gt = read_gt()
    rows = read_marker_rows()
    if not rows:
        raise ValueError("No blind visual_ihf marker rows found.")
    detail = compute_distances(rows, gt)
    emoji_images = read_html_emoji_images()
    participants, counts = validate_counts(rows)

    fig5c_canvas = draw_fig5c_png(rows, detail, gt, emoji_images)
    heatmap_canvas = draw_heatmap_png(rows, gt, emoji_images)

    outputs = []
    outputs.extend(save_bitmap(fig5c_canvas, FIG5C_STEM))
    outputs.append(write_fig5c_svg(rows, detail, gt, emoji_images, FIG5C_STEM))
    outputs.append(write_fig5c_pdf(rows, detail, gt, emoji_images, FIG5C_STEM))

    outputs.extend(save_bitmap(heatmap_canvas, HEATMAP_STEM))
    outputs.append(write_heatmap_svg(rows, gt, emoji_images, HEATMAP_STEM))
    outputs.append(write_heatmap_pdf(rows, gt, emoji_images, HEATMAP_STEM))

    print("Wrote:")
    for path in outputs:
        print(f"  {path}")
    print(f"cm_per_px={CM_PER_PX:.8f}; participants={len(participants)} ({', '.join(participants)})")
    for key in sorted(counts):
        print(f"{key[0]} {key[1]} n={counts[key]}")


if __name__ == "__main__":
    main()
