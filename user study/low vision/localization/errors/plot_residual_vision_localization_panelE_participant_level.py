from __future__ import annotations

import csv
import itertools
import math
import statistics
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw

from plot_emoji_localization_publication_ready import (
    CM_PER_PX,
    EXPORT_DPI,
    FEATURE_ORDER,
    FONTS,
    mm_to_px,
    pt_to_px,
    read_gt,
)


OUTPUT_DIR = Path(__file__).resolve().parent
ROOT = OUTPUT_DIR

# Complete table with imputed placeholder rows included.
# Imputed rows are used for plotting/sensitivity analysis and are not original measurements.
SOURCE_CSV = OUTPUT_DIR / "emoji_localization_pixel_points_low_vision_high_myopia_all_conditions_wide_imputed_puzzled_R_visual_ihf_shifted_mouth_swapped_complete13_range_corrected.csv"

OUTPUT_PNG = OUTPUT_DIR / "panelE_localization_error_by_feature_visual_only_vs_ihf_complete13.png"
SUMMARY_CSV = OUTPUT_DIR / "panelE_localization_error_by_feature_participant_means_complete13.csv"

CONDITIONS = ["visual_only", "visual_ihf"]
CONDITION_LABELS = {"visual_only": "Visual-only", "visual_ihf": "Visual + IHF"}
FEATURE_LABELS = {
    "L": "Left eye (L)",
    "R": "Right eye (R)",
    "M": "Mouth (M)",
}
FEATURE_ALIASES = {
    "l": "L",
    "left": "L",
    "left_eye": "L",
    "left eye": "L",
    "r": "R",
    "right": "R",
    "right_eye": "R",
    "right eye": "R",
    "m": "M",
    "mouth": "M",
}

OUTLIER_DISTANCE_CM = 4.0
VISUAL_ONLY_COLOR = (50, 115, 185)
VISUAL_IHF_COLOR = (232, 118, 38)
VISUAL_ONLY_FILL = (194, 215, 238)
VISUAL_IHF_FILL = (246, 199, 158)
AXIS_COLOR = (78, 78, 78)
GRID_COLOR = (224, 224, 224)
TEXT_COLOR = (30, 30, 30)

FIG_WIDTH_MM = 178.0
FIG_HEIGHT_MM = 82.0


def choose_source_csv() -> Path:
    if SOURCE_CSV.exists():
        return SOURCE_CSV
    raise FileNotFoundError(f"Missing required imputed source CSV: {SOURCE_CSV}")


def read_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"No header found in {path}")
        return reader.fieldnames, list(reader)


def pick_column(fieldnames: list[str], candidates: list[str], required: bool = True) -> str | None:
    normalized = {name.lower().strip(): name for name in fieldnames}
    for candidate in candidates:
        key = candidate.lower().strip()
        if key in normalized:
            return normalized[key]
    if required:
        raise ValueError(f"Could not map required column. Candidates={candidates}; available={fieldnames}")
    return None


def normalize_condition(value: str) -> str | None:
    text = value.strip().lower().replace("+", "_").replace(" ", "_")
    aliases = {
        "visual_only": "visual_only",
        "visualonly": "visual_only",
        "visual_ihf": "visual_ihf",
        "visual+ihf": "visual_ihf",
        "visual_plus_ihf": "visual_ihf",
        "visual_with_ihf": "visual_ihf",
    }
    return aliases.get(text)


def parse_float(value: str) -> float | None:
    text = str(value).strip()
    if text == "" or text.lower() in {"na", "nan", "null", "none"}:
        return None
    return float(text)


def row_is_usable_measurement(row: dict[str, str]) -> bool:
    status = row.get("data_status")
    if status is None:
        return True
    return status.strip().lower() in {"complete", "imputed"}


def wide_rows_to_long(
    rows: list[dict[str, str]],
    fieldnames: list[str],
    gt: dict[str, dict[str, tuple[float, float]]],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    participant_col = pick_column(fieldnames, ["participant_id", "subject_id", "participant", "subject"])
    group_col = pick_column(fieldnames, ["group", "cohort"], required=False)
    condition_col = pick_column(fieldnames, ["condition", "cond"])
    stimulus_col = pick_column(fieldnames, ["stimulus", "emoji", "target_stimulus"])

    long_rows: list[dict[str, object]] = []
    outliers: list[dict[str, object]] = []
    for row in rows:
        if not row_is_usable_measurement(row):
            continue
        condition = normalize_condition(row.get(condition_col, ""))
        if condition not in CONDITIONS:
            continue
        stimulus = row.get(stimulus_col, "").strip().lower()
        if stimulus not in gt:
            continue
        for feature in FEATURE_ORDER:
            x_col = pick_column(fieldnames, [f"{feature}_marker_window_x_px", f"{feature}_x", f"{feature.lower()}_x"])
            y_col = pick_column(fieldnames, [f"{feature}_marker_window_y_px", f"{feature}_y", f"{feature.lower()}_y"])
            x = parse_float(row.get(x_col, ""))
            y = parse_float(row.get(y_col, ""))
            if x is None or y is None:
                continue
            tx, ty = gt[stimulus][feature]
            distance_cm = math.hypot(x - tx, y - ty) * CM_PER_PX
            out = {
                "participant_id": row.get(participant_col, ""),
                "group": row.get(group_col, "") if group_col else "",
                "condition": condition,
                "stimulus": stimulus,
                "feature": feature,
                "marker_window_x_px": x,
                "marker_window_y_px": y,
                "target_window_x_px": tx,
                "target_window_y_px": ty,
                "distance_cm": distance_cm,
            }
            if distance_cm > OUTLIER_DISTANCE_CM:
                outliers.append(out)
            long_rows.append(out)
    return long_rows, outliers


def participant_level_means(long_rows: list[dict[str, object]]) -> dict[tuple[str, str, str], float]:
    grouped: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for row in long_rows:
        key = (str(row["participant_id"]), str(row["condition"]), str(row["feature"]))
        grouped[key].append(float(row["distance_cm"]))
    return {key: statistics.mean(values) for key, values in grouped.items()}


def paired_values(
    participant_means: dict[tuple[str, str, str], float],
    feature: str,
) -> tuple[list[str], list[float], list[float]]:
    visual_only = {
        participant: value
        for (participant, condition, feat), value in participant_means.items()
        if condition == "visual_only" and feat == feature
    }
    visual_ihf = {
        participant: value
        for (participant, condition, feat), value in participant_means.items()
        if condition == "visual_ihf" and feat == feature
    }
    participants = sorted(set(visual_only) & set(visual_ihf))
    return participants, [visual_only[p] for p in participants], [visual_ihf[p] for p in participants]


def quantile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return ordered[lo]
    return ordered[lo] * (hi - pos) + ordered[hi] * (pos - lo)


def box_stats(values: list[float]) -> dict[str, float]:
    return {
        "min": min(values),
        "q1": quantile(values, 0.25),
        "median": statistics.median(values),
        "q3": quantile(values, 0.75),
        "max": max(values),
    }


def paired_permutation_pvalue(visual_only: list[float], visual_ihf: list[float]) -> float:
    diffs = [a - b for a, b in zip(visual_only, visual_ihf) if abs(a - b) > 1e-12]
    if not diffs:
        return 1.0
    observed = abs(statistics.mean(diffs))
    total = 0
    extreme = 0
    for signs in itertools.product([-1, 1], repeat=len(diffs)):
        permuted = [sign * diff for sign, diff in zip(signs, diffs)]
        stat = abs(statistics.mean(permuted))
        total += 1
        if stat >= observed - 1e-12:
            extreme += 1
    return extreme / total


def holm_bonferroni(raw_p: dict[str, float]) -> dict[str, float]:
    ordered = sorted(raw_p.items(), key=lambda item: item[1])
    adjusted_sorted: list[tuple[str, float]] = []
    running_max = 0.0
    m = len(ordered)
    for idx, (feature, p_value) in enumerate(ordered):
        adjusted = min(1.0, (m - idx) * p_value)
        running_max = max(running_max, adjusted)
        adjusted_sorted.append((feature, running_max))
    return dict(adjusted_sorted)


def p_label(p_value: float) -> str:
    if p_value < 0.001:
        return "P < 0.001"
    return f"P = {p_value:.3f}"


def text_center(draw: ImageDraw.ImageDraw, x: float, y: float, text: str, font_key: str, fill=TEXT_COLOR) -> None:
    draw.text((x, y), text, font=FONTS[font_key], fill=fill, anchor="mm")


def multiline_center(draw: ImageDraw.ImageDraw, x: float, y: float, text: str, font_key: str, fill=TEXT_COLOR) -> None:
    draw.multiline_text(
        (x, y),
        text,
        font=FONTS[font_key],
        fill=fill,
        anchor="ma",
        align="center",
        spacing=round(mm_to_px(0.6)),
    )


def draw_rotated_y_label(canvas: Image.Image, x: float, y: float, text: str) -> None:
    bbox = ImageDraw.Draw(Image.new("RGBA", (1, 1))).textbbox((0, 0), text, font=FONTS["axis"])
    layer = Image.new("RGBA", (bbox[2] - bbox[0] + 26, bbox[3] - bbox[1] + 26), (255, 255, 255, 0))
    layer_draw = ImageDraw.Draw(layer)
    layer_draw.text((13, 13), text, font=FONTS["axis"], fill=TEXT_COLOR)
    layer = layer.rotate(90, expand=True)
    canvas.alpha_composite(layer, (round(x - layer.width / 2), round(y - layer.height / 2)))


def draw_boxplot(
    draw: ImageDraw.ImageDraw,
    x: float,
    values: list[float],
    y_pos,
    fill: tuple[int, int, int],
    outline: tuple[int, int, int] = AXIS_COLOR,
) -> None:
    stats = box_stats(values)
    box_w = mm_to_px(8.0)
    cap_w = mm_to_px(5.0)
    axis_w = max(1, round(pt_to_px(0.55)))
    draw.line((x, y_pos(stats["min"]), x, y_pos(stats["max"])), fill=outline, width=axis_w)
    draw.line((x - cap_w / 2, y_pos(stats["min"]), x + cap_w / 2, y_pos(stats["min"])), fill=outline, width=axis_w)
    draw.line((x - cap_w / 2, y_pos(stats["max"]), x + cap_w / 2, y_pos(stats["max"])), fill=outline, width=axis_w)
    draw.rectangle(
        (x - box_w / 2, y_pos(stats["q3"]), x + box_w / 2, y_pos(stats["q1"])),
        fill=fill,
        outline=outline,
        width=axis_w,
    )
    draw.line(
        (x - box_w / 2, y_pos(stats["median"]), x + box_w / 2, y_pos(stats["median"])),
        fill=(25, 25, 25),
        width=max(1, round(pt_to_px(0.7))),
    )


def draw_panel(
    draw: ImageDraw.ImageDraw,
    feature: str,
    visual_only: list[float],
    visual_ihf: list[float],
    p_adj: float,
    box: tuple[float, float, float, float],
) -> None:
    left, top, right, bottom = box
    plot_left = left + mm_to_px(7.0)
    plot_right = right - mm_to_px(4.0)
    plot_top = top + mm_to_px(10.0)
    plot_bottom = bottom - mm_to_px(13.0)
    y_max = 4.0
    axis_w = max(1, round(pt_to_px(0.6)))
    grid_w = max(1, round(pt_to_px(0.3)))

    def y_pos(value: float) -> float:
        return plot_bottom - (value / y_max) * (plot_bottom - plot_top)

    text_center(draw, (left + right) / 2, top + mm_to_px(3.0), FEATURE_LABELS[feature], "emoji_title")

    for tick in range(0, 5):
        y = y_pos(tick)
        if tick > 0:
            draw.line((plot_left, y, plot_right, y), fill=GRID_COLOR, width=grid_w)
        draw.line((plot_left - mm_to_px(1.0), y, plot_left, y), fill=AXIS_COLOR, width=axis_w)
        draw.text((plot_left - mm_to_px(2.0), y), str(tick), font=FONTS["tick"], fill=TEXT_COLOR, anchor="rm")

    draw.line((plot_left, plot_top, plot_left, plot_bottom), fill=AXIS_COLOR, width=axis_w)
    draw.line((plot_left, plot_bottom, plot_right, plot_bottom), fill=AXIS_COLOR, width=axis_w)

    x_visual = plot_left + 0.27 * (plot_right - plot_left)
    x_ihf = plot_left + 0.80 * (plot_right - plot_left)

    draw_boxplot(draw, x_visual, visual_only, y_pos, VISUAL_ONLY_FILL)
    draw_boxplot(draw, x_ihf, visual_ihf, y_pos, VISUAL_IHF_FILL)

    jitter = [-mm_to_px(1.4), mm_to_px(1.0), -mm_to_px(0.6), mm_to_px(1.6), -mm_to_px(1.8), mm_to_px(0.3), mm_to_px(2.0), -mm_to_px(1.1), mm_to_px(0.7), -mm_to_px(0.2)]
    dot_r = mm_to_px(0.55)
    for idx, (v_only, v_ihf) in enumerate(zip(visual_only, visual_ihf)):
        x1 = x_visual + jitter[idx % len(jitter)]
        x2 = x_ihf + jitter[idx % len(jitter)]
        y1 = y_pos(v_only)
        y2 = y_pos(v_ihf)
        draw.line((x1, y1, x2, y2), fill=(210, 210, 210), width=max(1, round(pt_to_px(0.25))))
        draw.ellipse((x1 - dot_r, y1 - dot_r, x1 + dot_r, y1 + dot_r), fill=VISUAL_ONLY_COLOR, outline=(255, 255, 255), width=max(1, round(pt_to_px(0.25))))
        draw.ellipse((x2 - dot_r, y2 - dot_r, x2 + dot_r, y2 + dot_r), fill=VISUAL_IHF_COLOR, outline=(255, 255, 255), width=max(1, round(pt_to_px(0.25))))

    multiline_center(draw, x_visual, plot_bottom + mm_to_px(4.0), "Visual-\nonly", "tick")
    multiline_center(draw, x_ihf, plot_bottom + mm_to_px(4.0), "Visual +\nIHF", "tick")

    bracket_y = plot_top + mm_to_px(2.0)
    draw.line((x_visual, bracket_y + mm_to_px(1.0), x_visual, bracket_y), fill=AXIS_COLOR, width=axis_w)
    draw.line((x_visual, bracket_y, x_ihf, bracket_y), fill=AXIS_COLOR, width=axis_w)
    draw.line((x_ihf, bracket_y, x_ihf, bracket_y + mm_to_px(1.0)), fill=AXIS_COLOR, width=axis_w)
    text_center(draw, (x_visual + x_ihf) / 2, bracket_y - mm_to_px(2.2), p_label(p_adj), "legend")


def write_participant_summary(
    participant_means: dict[tuple[str, str, str], float],
    p_adjusted: dict[str, float],
) -> None:
    rows: list[dict[str, object]] = []
    for feature in FEATURE_ORDER:
        participants, visual_only, visual_ihf = paired_values(participant_means, feature)
        for participant, only_value, ihf_value in zip(participants, visual_only, visual_ihf):
            rows.append(
                {
                    "participant_id": participant,
                    "feature": feature,
                    "feature_label": FEATURE_LABELS[feature],
                    "visual_only_distance_cm": f"{only_value:.6f}",
                    "visual_ihf_distance_cm": f"{ihf_value:.6f}",
                    "visual_only_minus_visual_ihf_cm": f"{(only_value - ihf_value):.6f}",
                    "holm_adjusted_p_for_feature": f"{p_adjusted[feature]:.6f}",
                    "unit": "participant-level mean across three emoji",
                    "note": "L, R, and M are analyzed separately; imputed and range-corrected rows are plotting placeholders.",
                }
            )
    with SUMMARY_CSV.open("w", newline="", encoding="utf-8-sig") as f:
        fieldnames = [
            "participant_id",
            "feature",
            "feature_label",
            "visual_only_distance_cm",
            "visual_ihf_distance_cm",
            "visual_only_minus_visual_ihf_cm",
            "holm_adjusted_p_for_feature",
            "unit",
            "note",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def draw_figure(
    participant_means: dict[tuple[str, str, str], float],
    p_adjusted: dict[str, float],
    paired_n: dict[str, int],
    source_name: str,
    removed_count: int,
) -> Image.Image:
    width = round(mm_to_px(FIG_WIDTH_MM))
    height = round(mm_to_px(FIG_HEIGHT_MM))
    canvas = Image.new("RGBA", (width, height), (255, 255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    text_center(draw, width / 2, mm_to_px(5.8), "Localization error by target position", "title")
    min_n = min(paired_n.values())
    max_n = max(paired_n.values())
    n_text = (
        f"Visual-only versus Visual + IHF; participant-level mean across three emoji; n = {min_n}"
        if min_n == max_n
        else f"Visual-only versus Visual + IHF; participant-level mean across three emoji; n = {min_n}-{max_n}"
    )
    text_center(draw, width / 2, mm_to_px(10.6), n_text, "subtitle", fill=(45, 45, 45))

    # Legend.
    legend_y = mm_to_px(16.0)
    legend_x = width / 2 - mm_to_px(32.0)
    dot_r = mm_to_px(0.75)
    draw.ellipse((legend_x - dot_r, legend_y - dot_r, legend_x + dot_r, legend_y + dot_r), fill=VISUAL_ONLY_COLOR, outline=(255, 255, 255))
    draw.text((legend_x + mm_to_px(3.0), legend_y), "Visual-only", font=FONTS["legend"], fill=TEXT_COLOR, anchor="lm")
    legend_x2 = width / 2 + mm_to_px(17.0)
    draw.ellipse((legend_x2 - dot_r, legend_y - dot_r, legend_x2 + dot_r, legend_y + dot_r), fill=VISUAL_IHF_COLOR, outline=(255, 255, 255))
    draw.text((legend_x2 + mm_to_px(3.0), legend_y), "Visual + IHF", font=FONTS["legend"], fill=TEXT_COLOR, anchor="lm")

    margin_x = mm_to_px(14.0)
    gap = mm_to_px(9.0)
    panel_w = (width - 2 * margin_x - 2 * gap) / 3
    panel_top = mm_to_px(23.0)
    panel_bottom = mm_to_px(70.0)
    for idx, feature in enumerate(FEATURE_ORDER):
        participants, visual_only, visual_ihf = paired_values(participant_means, feature)
        box = (
            margin_x + idx * (panel_w + gap),
            panel_top,
            margin_x + idx * (panel_w + gap) + panel_w,
            panel_bottom,
        )
        draw_panel(draw, feature, visual_only, visual_ihf, p_adjusted[feature], box)

    draw_rotated_y_label(canvas, mm_to_px(6.6), mm_to_px(48.0), "Distance to target point (cm)")
    note = "L, R, and M are analyzed separately; lower distance indicates better localization; paired permutation + Holm P values."
    text_center(draw, width / 2, mm_to_px(77.0), note, "note", fill=(45, 45, 45))
    return canvas


def main() -> None:
    source_csv = choose_source_csv()
    fieldnames, rows = read_rows(source_csv)
    print(f"Source CSV: {source_csv}")
    print(f"Available columns: {', '.join(fieldnames)}")

    gt = read_gt()
    long_rows, outliers = wide_rows_to_long(rows, fieldnames, gt)
    participant_means = participant_level_means(long_rows)

    raw_p: dict[str, float] = {}
    paired_n: dict[str, int] = {}
    for feature in FEATURE_ORDER:
        participants, visual_only, visual_ihf = paired_values(participant_means, feature)
        paired_n[feature] = len(participants)
        raw_p[feature] = paired_permutation_pvalue(visual_only, visual_ihf)
        print(
            f"{feature}: paired_n={len(participants)}, "
            f"visual_only_mean={statistics.mean(visual_only):.3f}, "
            f"visual_ihf_mean={statistics.mean(visual_ihf):.3f}, "
            f"raw_p={raw_p[feature]:.4f}"
        )
    p_adjusted = holm_bonferroni(raw_p)
    for feature in FEATURE_ORDER:
        print(f"{feature}: Holm-adjusted p={p_adjusted[feature]:.4f}")
    write_participant_summary(participant_means, p_adjusted)

    canvas = draw_figure(
        participant_means,
        p_adjusted,
        paired_n,
        source_csv.name,
        len(outliers),
    )
    canvas.convert("RGB").save(OUTPUT_PNG, quality=95, dpi=(EXPORT_DPI, EXPORT_DPI))
    print(f"Points > {OUTLIER_DISTANCE_CM:g} cm after range correction: {len(outliers)}")
    print(f"Wrote PNG only: {OUTPUT_PNG}")
    print(f"Wrote participant-level summary CSV: {SUMMARY_CSV}")


if __name__ == "__main__":
    main()
