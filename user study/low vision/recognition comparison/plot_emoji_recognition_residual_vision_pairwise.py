from __future__ import annotations

import csv
import itertools
import statistics
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


OUTPUT_DIR = Path(__file__).resolve().parent
ROOT = OUTPUT_DIR
TRIALS_CSV = OUTPUT_DIR / "emoji_confusion_matrix_trials_used.csv"
PARTICIPANT_CSV = OUTPUT_DIR / "emoji_recognition_accuracy_residual_vision_participant_level.csv"
GROUP_SUMMARY_CSV = OUTPUT_DIR / "emoji_recognition_accuracy_residual_vision_group_summary.csv"
OUTPUT_PNG = OUTPUT_DIR / "emoji_recognition_residual_vision_visual_only_vs_ihf_participant_level.png"

EXPORT_DPI = 600
FIG_WIDTH_MM = 178.0
FIG_HEIGHT_MM = 160.0
TARGET_GROUP = "clinical_low_vision_high_myopia"
CONDITIONS = ("visual_only", "visual_ihf")

BLUE = (28, 103, 190)
ORANGE = (255, 103, 0)
TEXT = (20, 20, 20)
AXIS = (55, 55, 55)
LIGHT_LINE = (188, 188, 188)
COHORT_GRAY = (82, 82, 82)


def mm_to_px(mm: float) -> float:
    return mm / 25.4 * EXPORT_DPI


def pt_to_px(pt: float) -> float:
    return pt / 72.0 * EXPORT_DPI


def font_pt(size_pt: float, bold: bool = False) -> ImageFont.FreeTypeFont:
    size_px = max(1, round(pt_to_px(size_pt)))
    candidates = [
        r"C:\Windows\Fonts\arialbd.ttf" if bold else r"C:\Windows\Fonts\arial.ttf",
        r"C:\Windows\Fonts\calibrib.ttf" if bold else r"C:\Windows\Fonts\calibri.ttf",
    ]
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size_px)
        except OSError:
            continue
    return ImageFont.load_default()


FONTS = {
    "title": font_pt(18.0, True),
    "subtitle": font_pt(13.5, True),
    "subnote": font_pt(10.5, False),
    "axis": font_pt(11.0, False),
    "tick": font_pt(10.0, False),
    "legend": font_pt(10.0, False),
    "mean": font_pt(12.0, True),
    "p": font_pt(10.5, False),
    "note": font_pt(9.0, False),
}


def read_trials() -> list[dict[str, str]]:
    with TRIALS_CSV.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def is_correct(row: dict[str, str]) -> bool:
    return row["true_key"].strip() == row["response_key"].strip()


def participant_accuracy_rows(trials: list[dict[str, str]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str, str], dict[str, int]] = {}
    for row in trials:
        if row["group"] != TARGET_GROUP or row["condition"] not in CONDITIONS:
            continue
        key = (row["participant_id"], row["cohort"], row["condition"])
        acc = grouped.setdefault(key, {"correct": 0, "total": 0})
        acc["total"] += 1
        acc["correct"] += int(is_correct(row))

    participants = sorted({participant for participant, _, _ in grouped})
    out: list[dict[str, object]] = []
    for participant in participants:
        cohort = next(cohort for pid, cohort, _ in grouped if pid == participant)
        row: dict[str, object] = {"participant_id": participant, "cohort": cohort}
        complete = True
        for condition in CONDITIONS:
            counts = grouped.get((participant, cohort, condition), {"correct": 0, "total": 0})
            total = counts["total"]
            accuracy = counts["correct"] / total * 100 if total else None
            row[f"{condition}_correct"] = counts["correct"]
            row[f"{condition}_total"] = total
            row[f"{condition}_accuracy_percent"] = accuracy
            complete = complete and total > 0
        if complete:
            row["visual_ihf_minus_visual_only_percent"] = (
                float(row["visual_ihf_accuracy_percent"]) - float(row["visual_only_accuracy_percent"])
            )
            out.append(row)
    return out


def paired_permutation_pvalue(visual_only: list[float], visual_ihf: list[float]) -> float:
    diffs = [ihf - only for only, ihf in zip(visual_only, visual_ihf) if abs(ihf - only) > 1e-12]
    if not diffs:
        return 1.0
    observed = abs(statistics.mean(diffs))
    total = 0
    extreme = 0
    for signs in itertools.product((-1, 1), repeat=len(diffs)):
        stat = abs(statistics.mean([sign * diff for sign, diff in zip(signs, diffs)]))
        total += 1
        if stat >= observed - 1e-12:
            extreme += 1
    return extreme / total


def write_csv_outputs(rows: list[dict[str, object]], p_value: float) -> None:
    fieldnames = [
        "participant_id",
        "cohort",
        "visual_only_correct",
        "visual_only_total",
        "visual_only_accuracy_percent",
        "visual_ihf_correct",
        "visual_ihf_total",
        "visual_ihf_accuracy_percent",
        "visual_ihf_minus_visual_only_percent",
    ]
    with PARTICIPANT_CSV.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: (
                        f"{float(row[key]):.6f}"
                        if key.endswith("_percent") and row.get(key) is not None
                        else row.get(key, "")
                    )
                    for key in fieldnames
                }
            )

    visual_only = [float(row["visual_only_accuracy_percent"]) for row in rows]
    visual_ihf = [float(row["visual_ihf_accuracy_percent"]) for row in rows]
    summary = [
        {
            "group": TARGET_GROUP,
            "condition": "visual_only",
            "n_participants": len(rows),
            "mean_accuracy_percent": f"{statistics.mean(visual_only):.6f}",
            "median_accuracy_percent": f"{statistics.median(visual_only):.6f}",
            "paired_permutation_p_value": f"{p_value:.6f}",
        },
        {
            "group": TARGET_GROUP,
            "condition": "visual_ihf",
            "n_participants": len(rows),
            "mean_accuracy_percent": f"{statistics.mean(visual_ihf):.6f}",
            "median_accuracy_percent": f"{statistics.median(visual_ihf):.6f}",
            "paired_permutation_p_value": f"{p_value:.6f}",
        },
    ]
    with GROUP_SUMMARY_CSV.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)


def text_center(draw: ImageDraw.ImageDraw, x: float, y: float, text: str, font_key: str, fill=TEXT) -> None:
    draw.text((x, y), text, font=FONTS[font_key], fill=fill, anchor="mm")


def text_left(draw: ImageDraw.ImageDraw, x: float, y: float, text: str, font_key: str, fill=TEXT) -> None:
    draw.text((x, y), text, font=FONTS[font_key], fill=fill, anchor="lm")


def draw_rotated_label(canvas: Image.Image, x: float, y: float, text: str) -> None:
    font = FONTS["axis"]
    bbox = ImageDraw.Draw(Image.new("RGBA", (1, 1))).textbbox((0, 0), text, font=font)
    layer = Image.new("RGBA", (bbox[2] - bbox[0] + 30, bbox[3] - bbox[1] + 30), (255, 255, 255, 0))
    layer_draw = ImageDraw.Draw(layer)
    layer_draw.text((15, 15), text, font=font, fill=TEXT)
    layer = layer.rotate(90, expand=True)
    canvas.alpha_composite(layer, (round(x - layer.width / 2), round(y - layer.height / 2)))


def triangle_points(cx: float, cy: float, r: float) -> list[tuple[float, float]]:
    return [(cx, cy - r * 1.08), (cx - r * 1.05, cy + r * 0.92), (cx + r * 1.05, cy + r * 0.92)]


def draw_marker(
    draw: ImageDraw.ImageDraw,
    x: float,
    y: float,
    color: tuple[int, int, int],
    cohort: str,
    radius: float,
) -> None:
    outline_w = max(1, round(pt_to_px(0.35)))
    if cohort == "clinical_low_vision":
        pts = triangle_points(x, y, radius)
        draw.polygon(pts, fill=color)
        draw.line(pts + [pts[0]], fill=(255, 255, 255), width=outline_w, joint="curve")
    else:
        draw.ellipse(
            (x - radius, y - radius, x + radius, y + radius),
            fill=color,
            outline=(255, 255, 255),
            width=outline_w,
        )


def draw_legend(draw: ImageDraw.ImageDraw, width: float, y: float) -> None:
    r = mm_to_px(1.55)
    x = mm_to_px(33)
    draw.ellipse((x - r, y - r, x + r, y + r), fill=BLUE)
    text_left(draw, x + mm_to_px(5), y, "Visual-only", "legend")

    x += mm_to_px(43)
    draw.ellipse((x - r, y - r, x + r, y + r), fill=ORANGE)
    text_left(draw, x + mm_to_px(5), y, "Visual + IHF", "legend")

    x += mm_to_px(50)
    draw.line((x - mm_to_px(4), y, x + mm_to_px(4), y), fill=LIGHT_LINE, width=max(1, round(pt_to_px(1.2))))
    text_left(draw, x + mm_to_px(8), y, "Same participant", "legend")

    y2 = y + mm_to_px(9)
    x2 = width / 2 - mm_to_px(30)
    draw.polygon(triangle_points(x2, y2, r * 0.95), fill=COHORT_GRAY)
    text_left(draw, x2 + mm_to_px(5), y2, "Clinical low vision", "legend")

    x3 = width / 2 + mm_to_px(25)
    draw.ellipse((x3 - r, y2 - r, x3 + r, y2 + r), fill=COHORT_GRAY)
    text_left(draw, x3 + mm_to_px(5), y2, "High myopia", "legend")


def draw_figure(rows: list[dict[str, object]], p_value: float) -> Image.Image:
    width = round(mm_to_px(FIG_WIDTH_MM))
    height = round(mm_to_px(FIG_HEIGHT_MM))
    canvas = Image.new("RGBA", (width, height), (255, 255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    text_center(draw, width / 2, mm_to_px(9.5), "Emoji recognition under residual visual access", "title")
    text_center(draw, width / 2, mm_to_px(23.5), "Clinical low vision + high myopia", "subtitle")
    text_center(draw, width / 2, mm_to_px(30.0), f"(n = {len(rows)} participants)", "subnote")

    plot_left = mm_to_px(31)
    plot_right = mm_to_px(163)
    plot_top = mm_to_px(44)
    plot_bottom = mm_to_px(113)
    x_visual = mm_to_px(70)
    x_ihf = mm_to_px(125)
    y_max = 100.0

    def y_pos(value: float) -> float:
        return plot_bottom - (value / y_max) * (plot_bottom - plot_top)

    axis_w = max(1, round(pt_to_px(0.8)))
    grid_w = max(1, round(pt_to_px(0.35)))
    for tick in (0, 25, 50, 75, 100):
        y = y_pos(tick)
        draw.line((plot_left - mm_to_px(1.6), y, plot_left, y), fill=AXIS, width=axis_w)
        draw.text((plot_left - mm_to_px(3.0), y), str(tick), font=FONTS["tick"], fill=TEXT, anchor="rm")
    draw.line((plot_left, plot_top, plot_left, plot_bottom), fill=AXIS, width=axis_w)
    draw.line((plot_left, plot_bottom, plot_right, plot_bottom), fill=AXIS, width=axis_w)

    draw_rotated_label(canvas, mm_to_px(18.0), (plot_top + plot_bottom) / 2, "Accuracy (%)")

    jitter = [
        -mm_to_px(5.5),
        -mm_to_px(4.4),
        -mm_to_px(3.3),
        -mm_to_px(2.2),
        -mm_to_px(1.1),
        0,
        mm_to_px(1.1),
        mm_to_px(2.2),
        mm_to_px(3.3),
        mm_to_px(4.4),
        mm_to_px(5.5),
        -mm_to_px(0.55),
        mm_to_px(0.55),
    ]
    sorted_rows = sorted(rows, key=lambda row: (float(row["visual_only_accuracy_percent"]), row["participant_id"]))
    points: list[tuple[float, float, float, float, str]] = []
    for idx, row in enumerate(sorted_rows):
        offset = jitter[idx % len(jitter)]
        y1 = y_pos(float(row["visual_only_accuracy_percent"]))
        y2 = y_pos(float(row["visual_ihf_accuracy_percent"]))
        x1 = x_visual + offset
        x2 = x_ihf + offset
        points.append((x1, y1, x2, y2, str(row["cohort"])))
        draw.line((x1, y1, x2, y2), fill=LIGHT_LINE, width=max(1, round(pt_to_px(0.75))))

    marker_r = mm_to_px(1.35)
    for x1, y1, x2, y2, cohort in points:
        draw_marker(draw, x1, y1, BLUE, cohort, marker_r)
        draw_marker(draw, x2, y2, ORANGE, cohort, marker_r)

    visual_only_values = [float(row["visual_only_accuracy_percent"]) for row in rows]
    visual_ihf_values = [float(row["visual_ihf_accuracy_percent"]) for row in rows]
    mean_visual = statistics.mean(visual_only_values)
    mean_ihf = statistics.mean(visual_ihf_values)
    mean_w = mm_to_px(11)
    mean_wide = max(1, round(pt_to_px(1.2)))
    draw.line((x_visual - mean_w / 2, y_pos(mean_visual), x_visual + mean_w / 2, y_pos(mean_visual)), fill=BLUE, width=mean_wide)
    draw.line((x_ihf - mean_w / 2, y_pos(mean_ihf), x_ihf + mean_w / 2, y_pos(mean_ihf)), fill=ORANGE, width=mean_wide)

    draw.text((x_visual - mm_to_px(20), y_pos(mean_visual)), f"{mean_visual:.1f}%", font=FONTS["mean"], fill=BLUE, anchor="rm")
    draw.text((x_ihf + mm_to_px(20), y_pos(mean_ihf)), f"{mean_ihf:.1f}%", font=FONTS["mean"], fill=ORANGE, anchor="lm")

    text_center(draw, x_visual, plot_bottom + mm_to_px(8.0), "Visual-only", "axis")
    text_center(draw, x_ihf, plot_bottom + mm_to_px(8.0), "Visual + IHF", "axis")

    bracket_y = plot_top - mm_to_px(5.2)
    bracket_h = mm_to_px(2.2)
    draw.line((x_visual, bracket_y + bracket_h, x_visual, bracket_y), fill=AXIS, width=axis_w)
    draw.line((x_visual, bracket_y, x_ihf, bracket_y), fill=AXIS, width=axis_w)
    draw.line((x_ihf, bracket_y, x_ihf, bracket_y + bracket_h), fill=AXIS, width=axis_w)
    p_text = "P < 0.05" if p_value < 0.05 else f"P = {p_value:.3f}"
    text_center(draw, (x_visual + x_ihf) / 2, bracket_y - mm_to_px(3.0), p_text, "p")

    draw_legend(draw, width, mm_to_px(136))
    note = "Statistics at participant level. Accuracy is computed as correct / valid emoji recognition trials."
    text_center(draw, width / 2, mm_to_px(154), note, "note")
    return canvas


def main() -> None:
    trials = read_trials()
    rows = participant_accuracy_rows(trials)
    if not rows:
        raise ValueError("No residual-vision emoji recognition rows found.")
    visual_only = [float(row["visual_only_accuracy_percent"]) for row in rows]
    visual_ihf = [float(row["visual_ihf_accuracy_percent"]) for row in rows]
    p_value = paired_permutation_pvalue(visual_only, visual_ihf)
    write_csv_outputs(rows, p_value)
    canvas = draw_figure(rows, p_value)
    canvas.convert("RGB").save(OUTPUT_PNG, quality=95, dpi=(EXPORT_DPI, EXPORT_DPI))

    print(f"participants={len(rows)}")
    print(f"visual_only_mean={statistics.mean(visual_only):.3f}")
    print(f"visual_ihf_mean={statistics.mean(visual_ihf):.3f}")
    print(f"paired_permutation_p={p_value:.6f}")
    print(f"Wrote participant CSV: {PARTICIPANT_CSV}")
    print(f"Wrote group summary CSV: {GROUP_SUMMARY_CSV}")
    print(f"Wrote PNG: {OUTPUT_PNG}")


if __name__ == "__main__":
    main()
