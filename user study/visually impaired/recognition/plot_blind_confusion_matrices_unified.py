from __future__ import annotations

import csv
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


FIG6_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = FIG6_DIR.parent

# Canonical blind-posture trial exports.  Do not change these raw records: the
# label correction below is applied only when constructing the figure matrix.
POSTURE_INPUT_FILES = [Path(__file__).resolve().parent / 'P004_2026-06-24T12-03-06_trial_summary_posture_corrected.csv', Path(__file__).resolve().parent / 'P011_2026-06-25T10-44-32_trial_summary_posture_corrected.csv', Path(__file__).resolve().parent / 'P019_2026-06-26T22-29-20_trial_summary_posture_supplemented.csv']
EMOJI_COUNTS = FIG6_DIR / "emoji_confusion_matrix_counts.csv"

OUTPUT_DIR = FIG6_DIR / "output"
OUTPUT_POSTURE = OUTPUT_DIR / "blind_posture_confusion_unified.png"
OUTPUT_EMOJI = OUTPUT_DIR / "blind_emoji_confusion_unified.png"

DPI = 600
POSTURE_LABELS = ["S", "HU", "D", "Y", "HS", "G"]
POSTURE_KEYS = [
    "standing",
    "hands_up",
    "dancing",
    "yoga",
    "horse_stance",
    "gymnastics",
]
BLIND_PARTICIPANTS = {"P002", "P003", "P005", "P006", "P007", "P012", "P014"}
# Confirmed from the original blind-data annotation: these two posture labels
# were recorded in reverse.  Remap both true and predicted keys so the matrix
# remains a comparison of like-for-like corrected categories.
BLIND_POSTURE_LABEL_CORRECTION = {
    "hands_up": "gymnastics",
    "gymnastics": "hands_up",
}
EMOJI_KEYS = ["screaming", "puzzled", "impatient"]
EMOJI_LABELS = ["S", "P", "I"]

# Same restrained sequential blue family used elsewhere in Fig. 5.
CMAP = ["#f0f8ff", "#cfe2f3", "#9fc5e8", "#3d85c6"]
INK = (25, 25, 25, 255)
WHITE = (255, 255, 255, 255)
TRANSPARENT = (255, 255, 255, 0)


def mm_to_px(value: float) -> float:
    return value / 25.4 * DPI


def pt_to_px(value: float) -> int:
    return max(1, round(value / 72.0 * DPI))


def load_font(size_pt: float) -> ImageFont.FreeTypeFont:
    for path in (
        Path(r"C:\Windows\Fonts") / "arial.ttf",
        Path(r"C:\Windows\Fonts") / "Arial.ttf",
        Path("/System/Library/Fonts/Supplemental/Arial.ttf"),
        Path("/opt/X11/share/system_fonts/Supplemental/Arial.ttf"),
    ):
        if path.exists():
            return ImageFont.truetype(str(path), pt_to_px(size_pt))
    return ImageFont.load_default(size=pt_to_px(size_pt))


FONT_AXIS = load_font(7.0)
FONT_TICK = load_font(7.0)
FONT_POSTURE_CELL = load_font(6.0)
FONT_EMOJI_CELL = load_font(7.0)


def hex_to_rgb(value: str) -> tuple[int, int, int]:
    value = value.lstrip("#")
    return tuple(int(value[index : index + 2], 16) for index in (0, 2, 4))


def blend_color(value: int) -> tuple[int, int, int, int]:
    fraction = max(0.0, min(100.0, float(value))) / 100.0
    colors = [hex_to_rgb(color) for color in CMAP]
    if fraction >= 1.0:
        return (*colors[-1], 255)
    span = 1.0 / (len(colors) - 1)
    index = min(len(colors) - 2, int(fraction / span))
    local = (fraction - index * span) / span
    rgb = tuple(
        round(
            colors[index][channel]
            + (colors[index + 1][channel] - colors[index][channel]) * local
        )
        for channel in range(3)
    )
    return (*rgb, 255)


def largest_remainder_percentages(counts: list[int]) -> list[int]:
    """Round a count row to integer percentages that sum to exactly 100.

    Integer arithmetic avoids floating-point tie artefacts. Equal remainders are
    resolved from left to right, which is deterministic and does not alter counts.
    """
    total = sum(counts)
    if total <= 0:
        raise ValueError("A true-class row has no valid observations")

    floors = [(100 * count) // total for count in counts]
    remainders = [(100 * count) % total for count in counts]
    missing = 100 - sum(floors)
    order = sorted(range(len(counts)), key=lambda index: (-remainders[index], index))

    rounded = floors[:]
    for index in order[:missing]:
        rounded[index] += 1

    if sum(rounded) != 100:
        raise AssertionError(f"Rounded row does not sum to 100: {rounded}")
    return rounded


def transpose(matrix: list[list[int]]) -> list[list[int]]:
    """Transpose true-by-predicted rows for predicted-by-true display."""
    return [list(column) for column in zip(*matrix)]


def read_posture_counts() -> list[list[int]]:
    index = {key: position for position, key in enumerate(POSTURE_KEYS)}
    counts = [[0 for _ in POSTURE_KEYS] for _ in POSTURE_KEYS]
    observations = 0

    for path in POSTURE_INPUT_FILES:
        with path.open(newline="", encoding="utf-8-sig") as handle:
            for row in csv.DictReader(handle):
                if row.get("participant_id") not in BLIND_PARTICIPANTS:
                    continue
                if row.get("task") != "posture" or row.get("condition") != "visual_ihf":
                    continue
                if row.get("skipped", "").strip().lower() == "true":
                    continue

                raw_true = row.get("true_key", "")
                raw_response = row.get("response_key", "")
                if raw_true not in index or raw_response not in index:
                    raise ValueError(f"Unexpected posture key in {path}: {row}")

                true_key = BLIND_POSTURE_LABEL_CORRECTION.get(raw_true, raw_true)
                response_key = BLIND_POSTURE_LABEL_CORRECTION.get(
                    raw_response, raw_response
                )
                counts[index[true_key]][index[response_key]] += 1
                observations += 1

    expected = len(BLIND_PARTICIPANTS) * len(POSTURE_KEYS)
    if observations != expected:
        raise ValueError(
            f"Expected {expected} blind visual-IHF posture trials, found {observations}"
        )
    return counts


def read_emoji_counts() -> list[list[int]]:
    index = {key: position for position, key in enumerate(EMOJI_KEYS)}
    counts = [[0 for _ in EMOJI_KEYS] for _ in EMOJI_KEYS]
    found = 0

    with EMOJI_COUNTS.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            if row["group"] != "blind" or row["condition"] != "visual_ihf":
                continue
            true_index = index[row["true_key"]]
            response_index = index[row["response_key"]]
            counts[true_index][response_index] = int(row["count"])
            found += 1

    expected = len(EMOJI_KEYS) ** 2
    if found != expected:
        raise ValueError(f"Expected {expected} emoji cells, found {found}")
    return counts


def draw_rotated_text(
    canvas: Image.Image,
    text: str,
    x: float,
    y: float,
    font: ImageFont.FreeTypeFont,
) -> None:
    probe = ImageDraw.Draw(Image.new("RGBA", (1, 1), TRANSPARENT))
    bbox = probe.textbbox((0, 0), text, font=font)
    padding = round(mm_to_px(2.0))
    layer = Image.new(
        "RGBA",
        (bbox[2] - bbox[0] + padding, bbox[3] - bbox[1] + padding),
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


def draw_matrix(
    percentages: list[list[int]],
    labels: list[str],
    width_mm: float,
    height_mm: float,
    grid_box_mm: tuple[float, float, float, float],
    cell_font: ImageFont.FreeTypeFont,
) -> Image.Image:
    width = round(mm_to_px(width_mm))
    height = round(mm_to_px(height_mm))
    canvas = Image.new("RGBA", (width, height), TRANSPARENT)
    draw = ImageDraw.Draw(canvas)

    left, top, right, bottom = [mm_to_px(value) for value in grid_box_mm]
    size = len(labels)
    cell_width = (right - left) / size
    cell_height = (bottom - top) / size
    separator = max(1, pt_to_px(0.55))

    draw.text(
        ((left + right) / 2, mm_to_px(2.7)),
        "True label",
        font=FONT_AXIS,
        fill=INK,
        anchor="mm",
    )
    draw_rotated_text(
        canvas,
        "Predicted label",
        mm_to_px(2.6),
        (top + bottom) / 2,
        FONT_AXIS,
    )

    for index, label in enumerate(labels):
        draw.text(
            (left + (index + 0.5) * cell_width, top - mm_to_px(2.2)),
            label,
            font=FONT_TICK,
            fill=INK,
            anchor="mm",
        )
        draw.text(
            (left - mm_to_px(1.6), top + (index + 0.5) * cell_height),
            label,
            font=FONT_TICK,
            fill=INK,
            anchor="rm",
        )

    for row in range(size):
        for column in range(size):
            value = percentages[row][column]
            x0 = left + column * cell_width
            y0 = top + row * cell_height
            draw.rectangle(
                (x0, y0, x0 + cell_width, y0 + cell_height),
                fill=blend_color(value),
                outline=WHITE,
                width=separator,
            )
            # White type on the darkest cells preserves contrast at final size.
            text_color = WHITE if value >= 65 else INK
            draw.text(
                (x0 + cell_width / 2, y0 + cell_height / 2),
                f"{value}%",
                font=cell_font,
                fill=text_color,
                anchor="mm",
            )

    return canvas


def validate_matrix(
    name: str,
    counts: list[list[int]],
    percentages_by_true: list[list[int]],
    display_percentages: list[list[int]],
) -> None:
    if len(counts) != len(percentages_by_true):
        raise AssertionError(f"{name}: count/percentage row mismatch")
    for row_index, (count_row, percent_row) in enumerate(
        zip(counts, percentages_by_true)
    ):
        if sum(percent_row) != 100:
            raise AssertionError(
                f"{name} true-class row {row_index} sums to "
                f"{sum(percent_row)}, not 100"
            )
        print(
            f"{name} true class {row_index + 1}: counts={count_row}; "
            f"rounded={percent_row}; sum={sum(percent_row)}%"
        )

    column_count = len(display_percentages[0])
    for column_index in range(column_count):
        column_sum = sum(
            display_percentages[row_index][column_index]
            for row_index in range(len(display_percentages))
        )
        if column_sum != 100:
            raise AssertionError(
                f"{name} displayed true-label column {column_index} "
                f"sums to {column_sum}, not 100"
            )
        print(
            f"{name} displayed true-label column {column_index + 1}: "
            f"sum={column_sum}%"
        )


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    posture_counts = read_posture_counts()
    emoji_counts = read_emoji_counts()

    posture_percentages_by_true = [
        largest_remainder_percentages(row) for row in posture_counts
    ]
    emoji_percentages_by_true = [
        largest_remainder_percentages(row) for row in emoji_counts
    ]

    # Match the residual-vision panels: columns are true labels and rows are
    # predicted labels. Each displayed true-label column therefore totals 100%.
    posture_percentages = transpose(posture_percentages_by_true)
    emoji_percentages = transpose(emoji_percentages_by_true)

    validate_matrix(
        "posture",
        posture_counts,
        posture_percentages_by_true,
        posture_percentages,
    )
    validate_matrix(
        "emoji",
        emoji_counts,
        emoji_percentages_by_true,
        emoji_percentages,
    )

    posture_image = draw_matrix(
        posture_percentages,
        POSTURE_LABELS,
        width_mm=54.0,
        height_mm=54.0,
        # Square 39 x 39 mm grid; each of the 6 x 6 cells is square.
        grid_box_mm=(11.0, 12.0, 50.0, 51.0),
        cell_font=FONT_POSTURE_CELL,
    )
    emoji_image = draw_matrix(
        emoji_percentages,
        EMOJI_LABELS,
        width_mm=54.0,
        height_mm=48.0,
        grid_box_mm=(11.0, 12.0, 41.0, 42.0),
        cell_font=FONT_EMOJI_CELL,
    )

    posture_image.save(OUTPUT_POSTURE, format="PNG", dpi=(DPI, DPI))
    emoji_image.save(OUTPUT_EMOJI, format="PNG", dpi=(DPI, DPI))
    print(f"Saved: {OUTPUT_POSTURE}")
    print(f"Saved: {OUTPUT_EMOJI}")


if __name__ == "__main__":
    main()
