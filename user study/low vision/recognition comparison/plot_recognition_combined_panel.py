from __future__ import annotations

"""Build the combined Fig. 5D-style recognition panel.

The participant-level data preparation and statistical tests are imported directly
from the two original plotting scripts. This file only combines their results into
one compact, transparent, publication/PPT-ready panel.
"""

import statistics
import sys
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


SOURCE_DIR = Path(__file__).resolve().parent
ROOT = SOURCE_DIR
EMOJI_DIR = SOURCE_DIR
POSTURE_DIR = SOURCE_DIR
for source_dir in (EMOJI_DIR, POSTURE_DIR):
    if str(source_dir) not in sys.path:
        sys.path.insert(0, str(source_dir))

# ORIGINAL PIPELINE REUSE:
# These modules retain the original input paths, filtering rules, participant-level
# accuracy calculations, cohort assignments, and statistical-test implementations.
import plot_emoji_recognition_residual_vision_pairwise as emoji_source
import plot_posture_recognition_vi_high_myopia as posture_source


OUTPUT_DIR = SOURCE_DIR.parent
OUTPUT_PNG = OUTPUT_DIR / "D_recognition_improvement_combined.png"
EMOJI_STANDALONE_PNG = OUTPUT_DIR / "D1_emoji_recognition_standalone.png"
POSTURE_STANDALONE_PNG = OUTPUT_DIR / "D2_posture_recognition_standalone.png"

# LAYOUT-ONLY PARAMETERS: safe to tune later without changing any data or statistics.
DPI = 600
FIG_WIDTH_MM = 94.0
FIG_HEIGHT_MM = 72.0
PLOT_TOP_MM = 20.0
PLOT_BOTTOM_MM = 51.2
STANDALONE_WIDTH_MM = 48.0
STANDALONE_HEIGHT_MM = 42.0

# Preserve the existing blue/orange condition palette and grey paired trajectories.
BLUE = posture_source.BLUE
ORANGE = posture_source.ORANGE
PAIR_GREY = posture_source.GREY_LINE
INK = posture_source.INK
AXIS = posture_source.AXIS
COHORT_GREY = "#555555"
EMOJI_TITLE = "#6A3D9A"
POSTURE_TITLE = "#2E7D32"
TRANSPARENT = (255, 255, 255, 0)


def mm(value: float) -> float:
    return value / 25.4 * DPI


def pt(value: float) -> float:
    return value / 72.0 * DPI


def load_font(size_pt: float, bold: bool = False) -> ImageFont.FreeTypeFont:
    candidates = [
        Path(r"C:\Windows\Fonts\arialbd.ttf" if bold else r"C:\Windows\Fonts\arial.ttf"),
        Path(r"C:\Windows\Fonts\calibrib.ttf" if bold else r"C:\Windows\Fonts\calibri.ttf"),
    ]
    for path in candidates:
        if path.exists():
            return ImageFont.truetype(str(path), size=max(1, round(pt(size_pt))))
    return ImageFont.load_default()


# LAYOUT-ONLY TYPOGRAPHY: all values are final-size point sizes.
FONTS = {
    "panel_title": load_font(8.5, True),
    "subplot_title": load_font(7.2, True),
    "n": load_font(6.1, False),
    "axis": load_font(6.5, False),
    "tick": load_font(6.1, False),
    "mean": load_font(6.8, True),
    "p": load_font(6.2, False),
    "legend": load_font(5.8, False),
}


def center_text(
    draw: ImageDraw.ImageDraw,
    x: float,
    y: float,
    text: str,
    font_key: str,
    fill=INK,
) -> None:
    draw.text((x, y), text, font=FONTS[font_key], fill=fill, anchor="mm")


def left_text(
    draw: ImageDraw.ImageDraw,
    x: float,
    y: float,
    text: str,
    font_key: str,
    fill=INK,
) -> None:
    draw.text((x, y), text, font=FONTS[font_key], fill=fill, anchor="lm")


def draw_rotated_text(
    canvas: Image.Image,
    x: float,
    y: float,
    text: str,
    font_key: str,
) -> None:
    font = FONTS[font_key]
    probe = ImageDraw.Draw(Image.new("RGBA", (1, 1), TRANSPARENT))
    bbox = probe.textbbox((0, 0), text, font=font)
    layer = Image.new(
        "RGBA",
        (bbox[2] - bbox[0] + round(mm(3)), bbox[3] - bbox[1] + round(mm(3))),
        TRANSPARENT,
    )
    layer_draw = ImageDraw.Draw(layer)
    layer_draw.text(
        (layer.width / 2, layer.height / 2),
        text,
        font=font,
        fill=INK,
        anchor="mm",
    )
    layer = layer.rotate(90, expand=True)
    canvas.alpha_composite(
        layer,
        (round(x - layer.width / 2), round(y - layer.height / 2)),
    )


def triangle_points(x: float, y: float, radius: float) -> list[tuple[float, float]]:
    return [
        (x, y - radius),
        (x - radius, y + radius),
        (x + radius, y + radius),
    ]


def draw_marker(
    draw: ImageDraw.ImageDraw,
    x: float,
    y: float,
    color,
    cohort: str,
    radius: float,
) -> None:
    """Preserve triangle=clinical low vision and circle=high myopia."""
    outline_width = max(1, round(pt(0.35)))
    is_clinical = cohort in {"clinical_low_vision", "Clinical low vision"}
    if is_clinical:
        points = triangle_points(x, y, radius)
        draw.polygon(points, fill=color)
        draw.line(
            points + [points[0]],
            fill="white",
            width=outline_width,
            joint="curve",
        )
    else:
        draw.ellipse(
            (x - radius, y - radius, x + radius, y + radius),
            fill=color,
            outline="white",
            width=outline_width,
        )


def load_original_results():
    """Call the original scripts' data and statistics functions without alteration."""
    emoji_trials = emoji_source.read_trials()
    emoji_original_rows = emoji_source.participant_accuracy_rows(emoji_trials)
    emoji_visual = [
        float(row["visual_only_accuracy_percent"]) for row in emoji_original_rows
    ]
    emoji_ihf = [
        float(row["visual_ihf_accuracy_percent"]) for row in emoji_original_rows
    ]
    emoji_p_value = emoji_source.paired_permutation_pvalue(emoji_visual, emoji_ihf)

    # This normalization only harmonizes field names for the shared drawing function.
    # It does not recalculate, modify, or interpolate participant values.
    emoji_rows = [
        {
            "participant_id": row["participant_id"],
            "cohort": row["cohort"],
            "visual_only": float(row["visual_only_accuracy_percent"]),
            "visual_ihf": float(row["visual_ihf_accuracy_percent"]),
        }
        for row in emoji_original_rows
    ]

    posture_rows = posture_source.load_posture_accuracy()
    posture_p_label = posture_source.sign_test_label(posture_rows)

    if not emoji_rows or not posture_rows:
        raise ValueError("Recognition data are missing from one or both original pipelines.")
    if len(emoji_rows) != len(posture_rows):
        raise ValueError(
            "The two recognition panels contain different participant counts: "
            f"emoji={len(emoji_rows)}, posture={len(posture_rows)}"
        )
    return emoji_rows, emoji_p_value, posture_rows, posture_p_label


def emoji_p_label(p_value: float) -> str:
    """Retain the original emoji script's computed P-value display rule."""
    return "P < 0.05" if p_value < 0.05 else f"P = {p_value:.3f}"


def emoji_offsets(rows: list[dict[str, object]]) -> dict[str, tuple[float, float]]:
    """Scaled version of the original emoji script's paired fixed-offset layout."""
    offsets_mm = [-2.5, -2.0, -1.5, -1.0, -0.5, 0, 0.5, 1.0, 1.5, 2.0, 2.5, -0.25, 0.25]
    sorted_rows = sorted(
        rows,
        key=lambda row: (float(row["visual_only"]), str(row["participant_id"])),
    )
    return {
        str(row["participant_id"]): (mm(offsets_mm[index]), mm(offsets_mm[index]))
        for index, row in enumerate(sorted_rows)
    }


def posture_condition_offsets(
    rows: list[dict[str, object]], condition: str
) -> dict[str, float]:
    """Scaled version of the original posture script's value-wise jitter logic."""
    groups: dict[float, list[str]] = defaultdict(list)
    for row in rows:
        groups[round(float(row[condition]), 1)].append(str(row["participant_id"]))

    offsets: dict[str, float] = {}
    for participant_ids in groups.values():
        participant_ids = sorted(participant_ids)
        count = len(participant_ids)
        if count == 1:
            local_offsets = [0.0]
        else:
            span = min(mm(5.0), mm(0.8) * (count - 1))
            local_offsets = [
                -span / 2 + span * index / (count - 1)
                for index in range(count)
            ]
        offsets.update(dict(zip(participant_ids, local_offsets)))
    return offsets


def posture_offsets(rows: list[dict[str, object]]) -> dict[str, tuple[float, float]]:
    visual = posture_condition_offsets(rows, "visual_only")
    ihf = posture_condition_offsets(rows, "visual_ihf")
    return {
        str(row["participant_id"]): (
            visual[str(row["participant_id"])],
            ihf[str(row["participant_id"])],
        )
        for row in rows
    }


def draw_subplot(
    canvas: Image.Image,
    draw: ImageDraw.ImageDraw,
    rows: list[dict[str, object]],
    title: str,
    title_color,
    p_label: str,
    axis_left_mm: float,
    axis_right_mm: float,
    offsets: dict[str, tuple[float, float]],
    show_y_label: bool,
) -> None:
    """LAYOUT-ONLY shared renderer; all values come from the original pipelines."""
    left = mm(axis_left_mm)
    right = mm(axis_right_mm)
    top = mm(PLOT_TOP_MM)
    bottom = mm(PLOT_BOTTOM_MM)
    center = (left + right) / 2
    x_visual = center - mm(6.7)
    x_ihf = center + mm(6.7)

    def y_position(value: float) -> float:
        return bottom - value / 100.0 * (bottom - top)

    title_center = (left + right) / 2
    center_text(draw, title_center, mm(9.3), title, "subplot_title", title_color)
    center_text(
        draw,
        title_center,
        mm(12.4),
        f"(n = {len(rows)} participants)",
        "n",
    )

    axis_width = max(1, round(pt(0.55)))
    for tick in (0, 25, 50, 75, 100):
        y = y_position(float(tick))
        draw.line((left - mm(0.8), y, left, y), fill=AXIS, width=axis_width)
        draw.text(
            (left - mm(1.25), y),
            str(tick),
            font=FONTS["tick"],
            fill=INK,
            anchor="rm",
        )
    draw.line((left, top, left, bottom), fill=AXIS, width=axis_width)
    draw.line((left, bottom, right, bottom), fill=AXIS, width=axis_width)

    if show_y_label:
        draw_rotated_text(
            canvas,
            mm(2.1),
            (top + bottom) / 2,
            "Accuracy (%)",
            "axis",
        )

    # Participant pairings: same participant is connected before markers are drawn.
    points = []
    for row in rows:
        participant = str(row["participant_id"])
        offset_visual, offset_ihf = offsets[participant]
        x1 = x_visual + offset_visual
        x2 = x_ihf + offset_ihf
        y1 = y_position(float(row["visual_only"]))
        y2 = y_position(float(row["visual_ihf"]))
        points.append((x1, y1, x2, y2, str(row["cohort"])))
        draw.line(
            (x1, y1, x2, y2),
            fill=PAIR_GREY,
            width=max(1, round(pt(0.55))),
        )

    marker_radius = mm(1.2)
    for x1, y1, x2, y2, cohort in points:
        draw_marker(draw, x1, y1, BLUE, cohort, marker_radius)
        draw_marker(draw, x2, y2, ORANGE, cohort, marker_radius)

    visual_values = [float(row["visual_only"]) for row in rows]
    ihf_values = [float(row["visual_ihf"]) for row in rows]
    mean_visual = statistics.mean(visual_values)
    mean_ihf = statistics.mean(ihf_values)
    mean_half_width = mm(3.2)
    mean_line_width = max(1, round(pt(1.0)))
    y_mean_visual = y_position(mean_visual)
    y_mean_ihf = y_position(mean_ihf)
    draw.line(
        (x_visual - mean_half_width, y_mean_visual, x_visual + mean_half_width, y_mean_visual),
        fill=BLUE,
        width=mean_line_width,
    )
    draw.line(
        (x_ihf - mean_half_width, y_mean_ihf, x_ihf + mean_half_width, y_mean_ihf),
        fill=ORANGE,
        width=mean_line_width,
    )
    draw.text(
        (x_visual - mm(4.0), y_mean_visual - mm(0.35)),
        f"{mean_visual:.1f}%",
        font=FONTS["mean"],
        fill=BLUE,
        anchor="rm",
    )
    draw.text(
        (x_ihf + mm(5.1), y_mean_ihf - mm(0.35)),
        f"{mean_ihf:.1f}%",
        font=FONTS["mean"],
        fill=ORANGE,
        anchor="lm",
    )

    center_text(draw, x_visual, bottom + mm(4.0), "Visual-only", "axis")
    center_text(draw, x_ihf, bottom + mm(4.0), "Visual + IHF", "axis")

    bracket_y = top - mm(2.2)
    bracket_height = mm(1.2)
    draw.line((x_visual, bracket_y, x_ihf, bracket_y), fill=AXIS, width=axis_width)
    draw.line(
        (x_visual, bracket_y, x_visual, bracket_y + bracket_height),
        fill=AXIS,
        width=axis_width,
    )
    draw.line(
        (x_ihf, bracket_y, x_ihf, bracket_y + bracket_height),
        fill=AXIS,
        width=axis_width,
    )
    center_text(
        draw,
        (x_visual + x_ihf) / 2,
        bracket_y - mm(1.45),
        p_label,
        "p",
    )


def draw_shared_legend(draw: ImageDraw.ImageDraw) -> None:
    """LAYOUT-ONLY compact, shared two-row legend for the full panel."""
    radius = mm(1.2)
    row1_y = mm(62.0)
    row2_y = mm(67.5)

    x = mm(6.5)
    draw_marker(draw, x, row1_y, BLUE, "High myopia", radius)
    left_text(draw, x + mm(2.3), row1_y, "Visual-only", "legend")

    x = mm(32.8)
    draw_marker(draw, x, row1_y, ORANGE, "High myopia", radius)
    left_text(draw, x + mm(2.3), row1_y, "Visual + IHF", "legend")

    line_left = mm(58.5)
    line_right = mm(65.0)
    draw.line(
        (line_left, row1_y, line_right, row1_y),
        fill=PAIR_GREY,
        width=max(1, round(pt(0.8))),
    )
    left_text(draw, line_right + mm(1.8), row1_y, "Same participant", "legend")

    x = mm(21.0)
    draw_marker(draw, x, row2_y, COHORT_GREY, "Clinical low vision", radius)
    left_text(draw, x + mm(2.3), row2_y, "Clinical low vision", "legend")

    x = mm(61.0)
    draw_marker(draw, x, row2_y, COHORT_GREY, "High myopia", radius)
    left_text(draw, x + mm(2.3), row2_y, "High myopia", "legend")


def draw_standalone_panel(
    rows: list[dict[str, object]],
    offsets: dict[str, tuple[float, float]],
) -> Image.Image:
    """Export a compact transparent plot without titles, n, P value, or legend.

    This is an additional layout-only export. It uses the same participant values,
    cohort marker encoding, condition colors, offsets, means, and 0-100% scale as the
    full panel. Only the essential axes, tick numbers, condition labels, and mean
    numbers are retained.
    """
    canvas = Image.new(
        "RGBA",
        (round(mm(STANDALONE_WIDTH_MM)), round(mm(STANDALONE_HEIGHT_MM))),
        TRANSPARENT,
    )
    draw = ImageDraw.Draw(canvas)
    left = mm(7.5)
    right = mm(46.5)
    top = mm(1.8)
    bottom = mm(35.0)
    x_visual = mm(19.5)
    x_ihf = mm(34.5)

    def y_position(value: float) -> float:
        return bottom - value / 100.0 * (bottom - top)

    axis_width = max(1, round(pt(0.55)))
    for tick in (0, 25, 50, 75, 100):
        y = y_position(float(tick))
        draw.line((left - mm(0.8), y, left, y), fill=AXIS, width=axis_width)
        draw.text(
            (left - mm(1.25), y),
            str(tick),
            font=FONTS["tick"],
            fill=INK,
            anchor="rm",
        )
    draw.line((left, top, left, bottom), fill=AXIS, width=axis_width)
    draw.line((left, bottom, right, bottom), fill=AXIS, width=axis_width)
    draw_rotated_text(
        canvas,
        mm(2.1),
        (top + bottom) / 2,
        "Accuracy (%)",
        "axis",
    )

    points = []
    for row in rows:
        participant = str(row["participant_id"])
        offset_visual, offset_ihf = offsets[participant]
        x1 = x_visual + offset_visual
        x2 = x_ihf + offset_ihf
        y1 = y_position(float(row["visual_only"]))
        y2 = y_position(float(row["visual_ihf"]))
        points.append((x1, y1, x2, y2, str(row["cohort"])))
        draw.line(
            (x1, y1, x2, y2),
            fill=PAIR_GREY,
            width=max(1, round(pt(0.55))),
        )

    marker_radius = mm(1.2)
    for x1, y1, x2, y2, cohort in points:
        draw_marker(draw, x1, y1, BLUE, cohort, marker_radius)
        draw_marker(draw, x2, y2, ORANGE, cohort, marker_radius)

    mean_visual = statistics.mean(float(row["visual_only"]) for row in rows)
    mean_ihf = statistics.mean(float(row["visual_ihf"]) for row in rows)
    mean_half_width = mm(3.2)
    mean_line_width = max(1, round(pt(1.0)))
    draw.line(
        (
            x_visual - mean_half_width,
            y_position(mean_visual),
            x_visual + mean_half_width,
            y_position(mean_visual),
        ),
        fill=BLUE,
        width=mean_line_width,
    )
    draw.line(
        (
            x_ihf - mean_half_width,
            y_position(mean_ihf),
            x_ihf + mean_half_width,
            y_position(mean_ihf),
        ),
        fill=ORANGE,
        width=mean_line_width,
    )
    draw.text(
        (x_visual - mm(4.0), y_position(mean_visual) - mm(0.35)),
        f"{mean_visual:.1f}%",
        font=FONTS["mean"],
        fill=BLUE,
        anchor="rm",
    )
    draw.text(
        (x_ihf + mm(5.1), y_position(mean_ihf) - mm(0.35)),
        f"{mean_ihf:.1f}%",
        font=FONTS["mean"],
        fill=ORANGE,
        anchor="lm",
    )
    center_text(draw, x_visual, bottom + mm(4.0), "Visual-only", "axis")
    center_text(draw, x_ihf, bottom + mm(4.0), "Visual + IHF", "axis")
    return canvas


def draw_figure(
    emoji_rows: list[dict[str, object]],
    emoji_p_value: float,
    posture_rows: list[dict[str, object]],
    posture_p_label: str,
) -> Image.Image:
    canvas = Image.new(
        "RGBA",
        (round(mm(FIG_WIDTH_MM)), round(mm(FIG_HEIGHT_MM))),
        TRANSPARENT,
    )
    draw = ImageDraw.Draw(canvas)

    center_text(
        draw,
        canvas.width / 2,
        mm(4.2),
        "Recognition improvement under residual visual access",
        "panel_title",
    )

    draw_subplot(
        canvas,
        draw,
        emoji_rows,
        "Emoji recognition",
        EMOJI_TITLE,
        emoji_p_label(emoji_p_value),
        7.0,
        44.0,
        emoji_offsets(emoji_rows),
        True,
    )
    draw_subplot(
        canvas,
        draw,
        posture_rows,
        "Posture recognition",
        POSTURE_TITLE,
        posture_p_label,
        50.0,
        91.0,
        posture_offsets(posture_rows),
        False,
    )
    draw_shared_legend(draw)
    return canvas


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    emoji_rows, emoji_p_value, posture_rows, posture_p_label = load_original_results()
    canvas = draw_figure(
        emoji_rows,
        emoji_p_value,
        posture_rows,
        posture_p_label,
    )
    canvas.save(OUTPUT_PNG, format="PNG", dpi=(DPI, DPI))
    draw_standalone_panel(emoji_rows, emoji_offsets(emoji_rows)).save(
        EMOJI_STANDALONE_PNG,
        format="PNG",
        dpi=(DPI, DPI),
    )
    draw_standalone_panel(posture_rows, posture_offsets(posture_rows)).save(
        POSTURE_STANDALONE_PNG,
        format="PNG",
        dpi=(DPI, DPI),
    )

    emoji_visual = [float(row["visual_only"]) for row in emoji_rows]
    emoji_ihf = [float(row["visual_ihf"]) for row in emoji_rows]
    posture_visual = [float(row["visual_only"]) for row in posture_rows]
    posture_ihf = [float(row["visual_ihf"]) for row in posture_rows]
    print(f"emoji_n={len(emoji_rows)}")
    print(f"emoji_visual_only_mean={statistics.mean(emoji_visual):.6f}")
    print(f"emoji_visual_ihf_mean={statistics.mean(emoji_ihf):.6f}")
    print(f"emoji_paired_permutation_p={emoji_p_value:.6f}")
    print(f"posture_n={len(posture_rows)}")
    print(f"posture_visual_only_mean={statistics.mean(posture_visual):.6f}")
    print(f"posture_visual_ihf_mean={statistics.mean(posture_ihf):.6f}")
    print(f"posture_sign_test_label={posture_p_label}")
    print(f"saved={OUTPUT_PNG}")
    print(f"saved={EMOJI_STANDALONE_PNG}")
    print(f"saved={POSTURE_STANDALONE_PNG}")


if __name__ == "__main__":
    main()
