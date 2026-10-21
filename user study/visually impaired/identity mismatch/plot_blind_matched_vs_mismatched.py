from __future__ import annotations

import csv
import itertools
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.stats import binomtest


# ---------------------------------------------------------------------------
# Figure contract
# ---------------------------------------------------------------------------
# Core conclusion: Correctly matched IHF improves recognition relative to the
# same participants receiving mismatched haptic mappings.
# Archetype: compact quantitative comparison panel.
# Export: transparent, 600 dpi PNG for insertion into a slide/PPT layout.

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
OUTPUT_DIR = SCRIPT_DIR / "output"
MISMATCH_DIR = SCRIPT_DIR

BLIND_PARTICIPANTS = ["P002", "P003", "P005", "P006", "P007", "P012", "P014"]
MATCHED_CONDITION = "visual_ihf"
MISMATCHED_CONDITION = "visual_mismatched_haptic"

EMOJI_SOURCE = SCRIPT_DIR / "emoji_confusion_matrix_trials_used.csv"
EMOJI_MISMATCH_SOURCE = (
    MISMATCH_DIR / "blind_emoji_recognition_mismatched_ihf_trials.csv"
)
POSTURE_MISMATCH_SOURCE = (
    MISMATCH_DIR / "blind_posture_recognition_mismatched_ihf_trials.csv"
)
POSTURE_MATCHED_SOURCES = [Path(__file__).resolve().parent / 'P004_2026-06-24T12-03-06_trial_summary_posture_corrected.csv', Path(__file__).resolve().parent / 'P011_2026-06-25T10-44-32_trial_summary_posture_corrected.csv', Path(__file__).resolve().parent / 'P019_2026-06-26T22-29-20_trial_summary_posture.csv']

EMOJI_CHANCE = 1 / 3
POSTURE_CHANCE = 1 / 6
BOOTSTRAP_REPLICATES = 200_000
RANDOM_SEED = 20260729


# ---------------------------------------------------------------------------
# Publication-oriented visual style
# ---------------------------------------------------------------------------

EXPORT_DPI = 600
FIGURE_WIDTH_MM = 117.8
FIGURE_HEIGHT_MM = 62.0

INK = "#202020"
AXIS = "#585858"
GRID = "#D9D9D9"
MATCHED = "#1769AA"
MATCHED_LIGHT = "#A9CAE1"
MISMATCHED = "#C9796C"
MISMATCHED_LIGHT = "#F1B8AC"
PAIR_LINE = "#B9B9B9"
CHANCE = "#E87500"
NOTE = "#707070"


def mm_to_px(value_mm: float) -> int:
    return round(value_mm / 25.4 * EXPORT_DPI)


def pt_to_px(value_pt: float) -> int:
    return max(1, round(value_pt / 72.0 * EXPORT_DPI))


def load_font(size_pt: float, bold: bool = False) -> ImageFont.FreeTypeFont:
    candidates = [
        Path(r"C:\Windows\Fonts\arialbd.ttf")
        if bold
        else Path(r"C:\Windows\Fonts\arial.ttf"),
        Path(r"C:\Windows\Fonts\calibrib.ttf")
        if bold
        else Path(r"C:\Windows\Fonts\calibri.ttf"),
    ]
    for path in candidates:
        if path.exists():
            return ImageFont.truetype(str(path), size=pt_to_px(size_pt))
    return ImageFont.load_default(size=pt_to_px(size_pt))


FONT_TITLE = load_font(8.2, True)
FONT_BODY = load_font(8.2)
FONT_BODY_BOLD = load_font(6.8, True)
FONT_TICK = load_font(8.2)
FONT_NOTE = load_font(8.0)


def text_width(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> int:
    box = draw.textbbox((0, 0), text, font=font)
    return box[2] - box[0]


def draw_centered(
    draw: ImageDraw.ImageDraw,
    x: float,
    y: float,
    text: str,
    font: ImageFont.ImageFont,
    fill: str = INK,
) -> None:
    draw.text((x - text_width(draw, text, font) / 2, y), text, font=font, fill=fill)


def draw_right(
    draw: ImageDraw.ImageDraw,
    x: float,
    y: float,
    text: str,
    font: ImageFont.ImageFont,
    fill: str = INK,
) -> None:
    draw.text((x - text_width(draw, text, font), y), text, font=font, fill=fill)


def draw_rotated_text(
    image: Image.Image,
    center: tuple[float, float],
    text: str,
    font: ImageFont.ImageFont,
    fill: str = INK,
) -> None:
    scratch = Image.new("RGBA", (pt_to_px(160), pt_to_px(18)), (255, 255, 255, 0))
    scratch_draw = ImageDraw.Draw(scratch)
    box = scratch_draw.textbbox((0, 0), text, font=font)
    scratch_draw.text((-box[0], -box[1]), text, font=font, fill=fill)
    crop = scratch.crop((0, 0, box[2] - box[0] + 4, box[3] - box[1] + 4))
    crop = crop.rotate(90, expand=True, resample=Image.Resampling.BICUBIC)
    image.alpha_composite(
        crop,
        (round(center[0] - crop.width / 2), round(center[1] - crop.height / 2)),
    )


def draw_circle(
    draw: ImageDraw.ImageDraw,
    center: tuple[float, float],
    radius: float,
    fill: str,
    outline: str | None = None,
    width: int = 1,
) -> None:
    x, y = center
    draw.ellipse(
        (x - radius, y - radius, x + radius, y + radius),
        fill=fill,
        outline=outline,
        width=width,
    )


def draw_dashed_line(
    draw: ImageDraw.ImageDraw,
    x1: float,
    x2: float,
    y: float,
    fill: str,
    width: int,
) -> None:
    dash = pt_to_px(3.1)
    gap = pt_to_px(2.0)
    cursor = x1
    while cursor < x2:
        end = min(x2, cursor + dash)
        draw.line((cursor, y, end, y), fill=fill, width=width)
        cursor = end + gap


# ---------------------------------------------------------------------------
# Data and statistics
# ---------------------------------------------------------------------------


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def participant_accuracies(rows: list[dict[str, str]]) -> dict[str, float]:
    result: dict[str, float] = {}
    for participant_id in BLIND_PARTICIPANTS:
        trials = [row for row in rows if row["participant_id"] == participant_id]
        if not trials:
            raise ValueError(f"No trials for {participant_id}.")
        result[participant_id] = sum(
            row["true_key"] == row["response_key"] for row in trials
        ) / len(trials)
    return result


def load_emoji(condition: str) -> list[dict[str, str]]:
    path = EMOJI_MISMATCH_SOURCE if condition == MISMATCHED_CONDITION else EMOJI_SOURCE
    rows = read_rows(path)
    return [
        row
        for row in rows
        if row.get("group") == "blind"
        and row.get("participant_id") in BLIND_PARTICIPANTS
        and row.get("condition") == condition
    ]


def load_posture(condition: str) -> list[dict[str, str]]:
    if condition == MISMATCHED_CONDITION:
        rows = read_rows(POSTURE_MISMATCH_SOURCE)
    else:
        rows = []
        for path in POSTURE_MATCHED_SOURCES:
            rows.extend(read_rows(path))
    return [
        row
        for row in rows
        if row.get("participant_id") in BLIND_PARTICIPANTS
        and row.get("task") == "posture"
        and row.get("condition") == condition
        and row.get("skipped", "").strip().lower() != "true"
    ]


def bootstrap_ci(values: np.ndarray, rng: np.random.Generator) -> tuple[float, float]:
    samples = rng.integers(0, len(values), size=(BOOTSTRAP_REPLICATES, len(values)))
    means = values[samples].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return float(low), float(high)


def exact_paired_signflip(differences: np.ndarray) -> float:
    signs = np.asarray(list(itertools.product((-1, 1), repeat=len(differences))))
    null_means = (signs * differences).mean(axis=1)
    observed = differences.mean()
    return float(np.count_nonzero(null_means >= observed - 1e-12) / len(null_means))


def summarize_task(
    task: str,
    matched_rows: list[dict[str, str]],
    mismatch_rows: list[dict[str, str]],
    chance: float,
    rng: np.random.Generator,
) -> dict[str, object]:
    matched = participant_accuracies(matched_rows)
    mismatch = participant_accuracies(mismatch_rows)
    matched_values = np.array([matched[pid] for pid in BLIND_PARTICIPANTS], dtype=float)
    mismatch_values = np.array([mismatch[pid] for pid in BLIND_PARTICIPANTS], dtype=float)
    differences = matched_values - mismatch_values
    difference_ci = bootstrap_ci(differences, rng)
    return {
        "task": task,
        "chance": chance,
        "matched": matched,
        "mismatch": mismatch,
        "matched_values": matched_values,
        "mismatch_values": mismatch_values,
        "matched_mean": float(matched_values.mean()),
        "mismatch_mean": float(mismatch_values.mean()),
        "matched_ci": bootstrap_ci(matched_values, rng),
        "mismatch_ci": bootstrap_ci(mismatch_values, rng),
        "difference_mean": float(differences.mean()),
        "difference_ci": difference_ci,
        "pair_p_raw": exact_paired_signflip(differences),
        "matched_n_correct": sum(
            row["true_key"] == row["response_key"] for row in matched_rows
        ),
        "matched_n_trials": len(matched_rows),
        "mismatch_n_correct": sum(
            row["true_key"] == row["response_key"] for row in mismatch_rows
        ),
        "mismatch_n_trials": len(mismatch_rows),
        "matched_chance_p_raw": float(
            binomtest(
                sum(row["true_key"] == row["response_key"] for row in matched_rows),
                len(matched_rows),
                chance,
                alternative="greater",
            ).pvalue
        ),
    }


def format_p(value: float) -> str:
    if value < 0.0001:
        return "P < 0.0001"
    return f"P = {value:.4f}".rstrip("0").rstrip(".")


# ---------------------------------------------------------------------------
# Plot rendering
# ---------------------------------------------------------------------------


def draw_bracket(
    draw: ImageDraw.ImageDraw,
    x1: float,
    x2: float,
    y: float,
    cap_height: float,
    width: int,
) -> None:
    draw.line((x1, y, x1, y + cap_height), fill=INK, width=width)
    draw.line((x1, y, x2, y), fill=INK, width=width)
    draw.line((x2, y, x2, y + cap_height), fill=INK, width=width)


def render_figure(summaries: list[dict[str, object]]) -> Image.Image:
    width = mm_to_px(FIGURE_WIDTH_MM)
    height = mm_to_px(FIGURE_HEIGHT_MM)
    image = Image.new("RGBA", (width, height), (255, 255, 255, 0))
    draw = ImageDraw.Draw(image)

    # Two separate task panels, matching the paired-comparison layout used for Fig. 5D.
    # Keeping independent axes avoids an oversized empty centre region and makes each
    # matched-versus-mismatch contrast readable at final manuscript size.
    plot_top = mm_to_px(7.2)
    plot_bottom = height - mm_to_px(14.5)
    y_max = 105.0

    def y_map(value: float) -> float:
        return plot_bottom - value / y_max * (plot_bottom - plot_top)

    axis_width = pt_to_px(0.55)
    participant_jitter = np.linspace(-pt_to_px(4.8), pt_to_px(4.8), len(BLIND_PARTICIPANTS))
    dot_radius = pt_to_px(2.0)

    # (left, right) positions in millimetres for Emoji and Posture panels.
    panel_bounds_mm = ((14.0, 53.0), (66.0, 112.0))

    for panel_index, (summary, (left_mm, right_mm)) in enumerate(
        zip(summaries, panel_bounds_mm)
    ):
        plot_left = mm_to_px(left_mm)
        plot_right = mm_to_px(right_mm)
        axis_span = plot_right - plot_left
        matched_x = plot_left + axis_span * 0.33
        mismatch_x = plot_left + axis_span * 0.73
        panel_center = (plot_left + plot_right) / 2

        draw.line(
            (plot_left, plot_top, plot_left, plot_bottom),
            fill=AXIS,
            width=axis_width,
        )
        draw.line(
            (plot_left, plot_bottom, plot_right, plot_bottom),
            fill=AXIS,
            width=axis_width,
        )
        for tick in (0, 25, 50, 75, 100):
            y = y_map(tick)
            draw.line(
                (plot_left - pt_to_px(2.0), y, plot_left, y),
                fill=AXIS,
                width=pt_to_px(0.45),
            )
            draw_right(
                draw,
                plot_left - pt_to_px(3.3),
                y - pt_to_px(3.2),
                str(tick),
                FONT_TICK,
                AXIS,
            )

            # Theoretical chance level
        chance_value = float(summary["chance"]) * 100.0
        chance_y = y_map(chance_value)

        dash_length = pt_to_px(3.0)
        dash_gap = pt_to_px(2.0)
        dash_x = plot_left

        while dash_x < plot_right:
            draw.line(
                (
                    dash_x,
                    chance_y,
                    min(dash_x + dash_length, plot_right),
                    chance_y,
                ),
                fill=CHANCE,
                width=pt_to_px(0.50),
            )
            dash_x += dash_length + dash_gap

        if panel_index == 0:
            draw_rotated_text(
                image,
                (mm_to_px(5.5), (plot_top + plot_bottom) / 2),
                "Accuracy (%)",
                FONT_BODY,
                INK,
            )

        for participant_id, jitter in zip(BLIND_PARTICIPANTS, participant_jitter):
            matched_value = float(summary["matched"][participant_id]) * 100
            mismatch_value = float(summary["mismatch"][participant_id]) * 100
            draw.line(
                (
                    matched_x + jitter,
                    y_map(matched_value),
                    mismatch_x + jitter,
                    y_map(mismatch_value),
                ),
                fill=PAIR_LINE,
                width=pt_to_px(0.55),
            )
            draw_circle(
                draw,
                (matched_x + jitter, y_map(matched_value)),
                dot_radius,
                MATCHED_LIGHT,
                MATCHED,
                pt_to_px(0.45),
            )
            draw_circle(
                draw,
                (mismatch_x + jitter, y_map(mismatch_value)),
                dot_radius,
                MISMATCHED_LIGHT,
                MISMATCHED,
                pt_to_px(0.45),
            )

        # Reference-style group means: a short horizontal line plus a direct label.
        matched_mean = float(summary["matched_mean"]) * 100
        mismatch_mean = float(summary["mismatch_mean"]) * 100
        mean_half_width = mm_to_px(3.2)
        mean_line_width = pt_to_px(1.0)
        matched_mean_y = y_map(matched_mean)
        mismatch_mean_y = y_map(mismatch_mean)
        draw.line(
            (matched_x - mean_half_width, matched_mean_y, matched_x + mean_half_width, matched_mean_y),
            fill=MATCHED,
            width=mean_line_width,
        )
        draw.line(
            (mismatch_x - mean_half_width, mismatch_mean_y, mismatch_x + mean_half_width, mismatch_mean_y),
            fill=MISMATCHED,
            width=mean_line_width,
        )
        draw_right(
            draw,
            matched_x - mm_to_px(4.0),
            matched_mean_y - mm_to_px(1.0),
            f"{matched_mean:.1f}%",
            FONT_BODY_BOLD,
            MATCHED,
        )
        draw.text(
            (mismatch_x + mm_to_px(4.0), mismatch_mean_y - mm_to_px(1.0)),
            f"{mismatch_mean:.1f}%",
            font=FONT_BODY_BOLD,
            fill=MISMATCHED,
        )

        # Show the exact unadjusted paired P value for each task.
        bracket_y = plot_top - mm_to_px(2.0)
        draw_bracket(
            draw,
            matched_x,
            mismatch_x,
            bracket_y,
            mm_to_px(1.2),
            axis_width,
        )
        p_label = format_p(float(summary["pair_p_raw"]))
        draw_centered(
            draw,
            panel_center,
            bracket_y - mm_to_px(3.1),
            p_label,
            FONT_NOTE,
            INK,
        )

        label_y = plot_bottom + mm_to_px(2.5)
        draw_centered(draw, matched_x, label_y, "Matched", FONT_BODY_BOLD, MATCHED)
        draw_centered(draw, mismatch_x, label_y, "Mismatch", FONT_BODY_BOLD, MISMATCHED)
        draw_centered(
            draw,
            panel_center,
            label_y + mm_to_px(5.5),
            str(summary["task"]),
            FONT_BODY_BOLD,
            INK,
        )
    return image


# ---------------------------------------------------------------------------
# Source-data and reporting files
# ---------------------------------------------------------------------------


def write_statistics(summaries: list[dict[str, object]]) -> None:
    path = OUTPUT_DIR / "figure6b_blind_matched_vs_mismatched_statistics.csv"
    fields = [
        "task",
        "n_participants",
        "chance_percent",
        "matched_correct_trials",
        "matched_total_trials",
        "matched_accuracy_percent",
        "matched_bootstrap_ci_low_percent",
        "matched_bootstrap_ci_high_percent",
        "mismatch_correct_trials",
        "mismatch_total_trials",
        "mismatch_accuracy_percent",
        "mismatch_bootstrap_ci_low_percent",
        "mismatch_bootstrap_ci_high_percent",
        "paired_difference_percentage_points",
        "paired_difference_bootstrap_ci_low_pp",
        "paired_difference_bootstrap_ci_high_pp",
        "paired_p_unadjusted",
        "matched_vs_chance_p_raw",
        "paired_test",
        "ci_definition",
    ]
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for summary in summaries:
            writer.writerow(
                {
                    "task": summary["task"],
                    "n_participants": len(BLIND_PARTICIPANTS),
                    "chance_percent": f"{float(summary['chance']) * 100:.6f}",
                    "matched_correct_trials": summary["matched_n_correct"],
                    "matched_total_trials": summary["matched_n_trials"],
                    "matched_accuracy_percent": f"{float(summary['matched_mean']) * 100:.6f}",
                    "matched_bootstrap_ci_low_percent": f"{float(summary['matched_ci'][0]) * 100:.6f}",
                    "matched_bootstrap_ci_high_percent": f"{float(summary['matched_ci'][1]) * 100:.6f}",
                    "mismatch_correct_trials": summary["mismatch_n_correct"],
                    "mismatch_total_trials": summary["mismatch_n_trials"],
                    "mismatch_accuracy_percent": f"{float(summary['mismatch_mean']) * 100:.6f}",
                    "mismatch_bootstrap_ci_low_percent": f"{float(summary['mismatch_ci'][0]) * 100:.6f}",
                    "mismatch_bootstrap_ci_high_percent": f"{float(summary['mismatch_ci'][1]) * 100:.6f}",
                    "paired_difference_percentage_points": f"{float(summary['difference_mean']) * 100:.6f}",
                    "paired_difference_bootstrap_ci_low_pp": f"{float(summary['difference_ci'][0]) * 100:.6f}",
                    "paired_difference_bootstrap_ci_high_pp": f"{float(summary['difference_ci'][1]) * 100:.6f}",
                    "paired_p_unadjusted": f"{float(summary['pair_p_raw']):.12g}",
                    "matched_vs_chance_p_raw": f"{float(summary['matched_chance_p_raw']):.12g}",
                    "paired_test": "One-sided exact paired sign-flip permutation test (matched > mismatch), unadjusted.",
                    "ci_definition": "Participant-level non-parametric bootstrap 95% confidence interval (200,000 resamples).",
                }
            )


def write_source_data(summaries: list[dict[str, object]]) -> None:
    path = OUTPUT_DIR / "figure6b_blind_matched_vs_mismatched_source_data.csv"
    fields = ["task", "participant_id", "matched_accuracy_percent", "mismatch_accuracy_percent", "difference_percentage_points"]
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for summary in summaries:
            for participant_id in BLIND_PARTICIPANTS:
                matched = float(summary["matched"][participant_id]) * 100
                mismatch = float(summary["mismatch"][participant_id]) * 100
                writer.writerow(
                    {
                        "task": summary["task"],
                        "participant_id": participant_id,
                        "matched_accuracy_percent": f"{matched:.6f}",
                        "mismatch_accuracy_percent": f"{mismatch:.6f}",
                        "difference_percentage_points": f"{matched - mismatch:.6f}",
                    }
                )


def write_caption(summaries: list[dict[str, object]]) -> None:
    emoji, posture = summaries
    caption = (
        "Fig. 6b | Recognition requires semantically matched IHF. Blue and pink dots "
        "denote individual blind participants (n = 7) under matched and mismatched "
        "IHF, respectively; lines connect the same participant. Coloured horizontal "
        "bars and adjacent labels denote participant-level group means. Matched IHF outperformed mismatched "
        f"IHF for emoji recognition by {float(emoji['difference_mean']) * 100:.1f} "
        f"percentage points (95% CI {float(emoji['difference_ci'][0]) * 100:.1f} to "
        f"{float(emoji['difference_ci'][1]) * 100:.1f}; paired "
        f"P = {float(emoji['pair_p_raw']):.4f}) and for posture recognition by "
        f"{float(posture['difference_mean']) * 100:.1f} percentage points (95% CI "
        f"{float(posture['difference_ci'][0]) * 100:.1f} to "
        f"{float(posture['difference_ci'][1]) * 100:.1f}; paired "
        f"P = {float(posture['pair_p_raw']):.4f}). Paired P values are from "
        "unadjusted one-sided exact sign-flip permutation tests (matched > mismatch)."
    )
    (OUTPUT_DIR / "figure6b_blind_matched_vs_mismatched_caption.txt").write_text(
        caption + "\n", encoding="utf-8"
    )


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(RANDOM_SEED)
    summaries = [
        summarize_task(
            "Emoji",
            load_emoji(MATCHED_CONDITION),
            load_emoji(MISMATCHED_CONDITION),
            EMOJI_CHANCE,
            rng,
        ),
        summarize_task(
            "Posture",
            load_posture(MATCHED_CONDITION),
            load_posture(MISMATCHED_CONDITION),
            POSTURE_CHANCE,
            rng,
        ),
    ]
    figure = render_figure(summaries)
    png_path = OUTPUT_DIR / "figure6b_blind_matched_vs_mismatched_recognition_transparent.png"
    figure.save(png_path, dpi=(EXPORT_DPI, EXPORT_DPI), optimize=True)
    write_statistics(summaries)
    write_source_data(summaries)
    write_caption(summaries)

    for summary in summaries:
        print(
            f"{summary['task']}: matched={float(summary['matched_mean']) * 100:.2f}%, "
            f"mismatch={float(summary['mismatch_mean']) * 100:.2f}%, "
            f"difference={float(summary['difference_mean']) * 100:.2f} pp "
            f"(95% CI {float(summary['difference_ci'][0]) * 100:.2f} to "
            f"{float(summary['difference_ci'][1]) * 100:.2f}); "
            f"paired p_unadjusted={float(summary['pair_p_raw']):.6g}"
        )
    print(f"Saved: {png_path}")


if __name__ == "__main__":
    main()
