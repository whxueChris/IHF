from pathlib import Path
import csv

from posture_confusion_style import render_confusion_matrix


ASSET_DIR = Path(__file__).resolve().parent
DATA_FILE = ASSET_DIR / "emoji_confusion_data" / "emoji_confusion_matrix_counts.csv"
OUTPUT_DIR = ASSET_DIR

GROUP = "clinical_low_vision_high_myopia"
EMOJI_ORDER = ["screaming", "puzzled", "impatient"]
EMOJI_LABELS = ["S", "P", "I"]
CONDITIONS = [
    ("visual_only", "visual_only"),
    ("visual_ihf", "visual_ihf"),
    ("visual_mismatched_haptic", "visual_mismatch_ihf"),
]


def load_matrix(condition):
    index = {emoji: position for position, emoji in enumerate(EMOJI_ORDER)}
    matrix = [[None for _ in EMOJI_ORDER] for _ in EMOJI_ORDER]
    row_totals = {}
    row_count_sums = {emoji: 0 for emoji in EMOJI_ORDER}

    with DATA_FILE.open(newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            if row["group"] != GROUP or row["condition"] != condition:
                continue
            true_key = row["true_key"]
            response_key = row["response_key"]
            count = int(row["count"])
            row_total = int(row["row_total"])
            if row_total <= 0:
                raise ValueError(f"Invalid row total for {condition}, {true_key}")
            if true_key in row_totals and row_totals[true_key] != row_total:
                raise ValueError(f"Inconsistent row total for {condition}, {true_key}")
            row_totals[true_key] = row_total
            row_count_sums[true_key] += count
            matrix[index[true_key]][index[response_key]] = count / row_total * 100.0

    if any(value is None for row in matrix for value in row):
        raise ValueError(f"Missing emoji confusion-matrix entries for {condition}")
    for true_key in EMOJI_ORDER:
        if row_count_sums[true_key] != row_totals[true_key]:
            raise ValueError(
                f"Counts do not sum to row_total for {condition}, {true_key}: "
                f"{row_count_sums[true_key]} != {row_totals[true_key]}"
            )
    return matrix


def main():
    for condition, file_suffix in CONDITIONS:
        matrix = load_matrix(condition)
        output_path = OUTPUT_DIR / f"residual_visual_access_emoji_confusion_{file_suffix}.png"
        render_confusion_matrix(
            matrix,
            EMOJI_LABELS,
            "Clinical low vision + high myopia",
            condition,
            output_path,
            # Compact 3 x 3 layout for the combined publication panel.
            # The percentage labels retain their original font size; only the
            # physical cell/canvas size is reduced. Exact values remain visible,
            # so the redundant per-matrix colorbar is omitted.
            image_size=(950, 950),
            grid_bounds=(140.0, 110.0, 800.0, 770.0),
            show_colorbar=False,
            force_column_sum_100=True,
        )
        print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
