from pathlib import Path
import csv
import contextlib
import io
import math

from posture_confusion_style import render_confusion_matrix


ASSET_DIR = Path(__file__).resolve().parent
DATA_DIR = ASSET_DIR / "posture_confusion_data"
OUTPUT_DIR = ASSET_DIR
DPI = 600

GROUPS = {
    "residual_visual_access": {
        "title": "Clinical low vision + high myopia",
        "participants": {
            "P001",
            "P004",
            "P008",
            "P009",
            "P010",
            "P011",
            "P013",
            "P015",
            "P016",
            "P017",
            "P018",
            "P019",
            "P020",
        },
    },
}

CONDITIONS = [
    ("visual_only", "Visual-only"),
    ("visual_ihf", "Visual + IHF"),
    ("visual_mismatched_haptic", "Visual + mismatch IHF"),
]

OUTPUT_FILES = {
    (group, condition): OUTPUT_DIR
    / f"{group}_posture_confusion_{condition.replace('visual_mismatched_haptic', 'visual_mismatch_ihf')}.png"
    for group in GROUPS
    for condition, _ in CONDITIONS
}

POSTURE_ORDER = [
    "standing",
    "hands_up",
    "dancing",
    "yoga",
    "horse_stance",
    "gymnastics",
]
POSTURE_LABELS = ["S", "HU", "D", "Y", "HS", "G"]
COLORS = ["#f0f8ff", "#cfe2f3", "#9fc5e8", "#3d85c6"]
INK = "#08142f"


def corrected_input_files():
    return [
        (
            "corrected_p001_p003_p004",
            DATA_DIR / "P004_2026-06-24T12-03-06_trial_summary_posture_corrected.csv",
        ),
        (
            "corrected_p002_p011",
            DATA_DIR / "P011_2026-06-25T10-44-32_trial_summary_posture_corrected.csv",
        ),
        (
            "corrected_p012_p019",
            DATA_DIR / "P019_2026-06-26T22-29-20_trial_summary_posture_corrected.csv",
        ),
        (
            "corrected_p020",
            DATA_DIR / "P020_2026-06-27T21-22-19_trial_summary_posture_corrected.csv",
        ),
    ]


def load_residual_posture_trials():
    trials = []
    for label, path in corrected_input_files():
        if not path.exists():
            raise FileNotFoundError(path)
        with path.open(newline="", encoding="utf-8-sig") as f:
            for line_no, row in enumerate(csv.DictReader(f), start=2):
                if row.get("task") != "posture":
                    continue
                if label == "corrected_p001_p003_p004":
                    if row["participant_id"] == "P001" and line_no < 59:
                        continue
                    if row["participant_id"] not in {"P001", "P003", "P004"}:
                        continue
                if not any(
                    row.get("participant_id", "") in config["participants"]
                    for config in GROUPS.values()
                ):
                    continue
                if row.get("condition") not in {condition for condition, _ in CONDITIONS}:
                    continue
                if row.get("skipped", "").strip().lower() == "true":
                    continue
                true_key = row.get("true_key", "")
                response_key = row.get("response_key", "")
                if true_key not in POSTURE_ORDER or response_key not in POSTURE_ORDER:
                    raise ValueError(f"Unexpected posture key in {path}: {row}")
                trials.append(row)
    return trials


def build_confusion_matrix(trials, group, condition):
    index = {label: i for i, label in enumerate(POSTURE_ORDER)}
    counts = [[0 for _ in POSTURE_ORDER] for _ in POSTURE_ORDER]

    for row in trials:
        if (
            row["participant_id"] not in GROUPS[group]["participants"]
            or row["condition"] != condition
        ):
            continue
        counts[index[row["true_key"]]][index[row["response_key"]]] += 1

    # Normalize each true-label row. The renderer transposes this matrix for
    # display (true labels on the x axis and predicted labels on the y axis),
    # so every displayed true-label column is a complete prediction
    # distribution that totals 100% before integer rounding.
    percent = [[0.0 for _ in POSTURE_ORDER] for _ in POSTURE_ORDER]
    for true_index in range(len(POSTURE_ORDER)):
        total = sum(counts[true_index][predicted_index] for predicted_index in range(len(POSTURE_ORDER)))
        if total == 0:
            raise ValueError(
                f"No trials were presented with true posture "
                f"{POSTURE_ORDER[true_index]!r} for {group}, {condition}"
            )
        for predicted_index in range(len(POSTURE_ORDER)):
            percent[true_index][predicted_index] = (
                counts[true_index][predicted_index] / total * 100
            )
        if not math.isclose(
            sum(percent[true_index][predicted_index] for predicted_index in range(len(POSTURE_ORDER))),
            100.0,
            abs_tol=1e-9,
        ):
            raise AssertionError("True-label row did not normalize to 100%")
    return counts, percent


def save_with_matplotlib(output_path, matrix):
    import numpy as np
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    from matplotlib.colors import LinearSegmentedColormap

    def heatmap(data, row_labels, col_labels, ax=None, cbar_kw={}, cbarlabel="", **kwargs):
        if not ax:
            ax = plt.gca()

        im = ax.imshow(data, **kwargs)
        cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
        cbar.ax.set_xlabel(cbarlabel, va="bottom", fontsize=15, family="Arial")

        ax.set_xticks(np.arange(data.shape[1]))
        ax.set_yticks(np.arange(data.shape[0]))
        ax.set_xticklabels(col_labels, fontsize=12, family="Arial")
        ax.set_yticklabels(row_labels, fontsize=12, family="Arial")

        ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)
        plt.setp(ax.get_xticklabels(), rotation=-0, ha="center", rotation_mode="anchor")

        for _, spine in ax.spines.items():
            spine.set_visible(False)

        ax.set_xticks(np.arange(data.shape[1] + 1) - 0.5, minor=True)
        ax.set_yticks(np.arange(data.shape[0] + 1) - 0.5, minor=True)
        ax.grid(which="minor", color="w", linestyle="-", linewidth=3)
        ax.tick_params(which="minor", bottom=False, left=False)

        formatter = ticker.FuncFormatter(lambda x, pos: f"{int(x)}%")
        cbar.ax.yaxis.set_major_formatter(formatter)
        cbar.ax.tick_params(labelsize=14)
        cbar.ax.tick_params(labelrotation=0)

        return im, cbar

    def annotate_heatmap(
        im,
        data=None,
        valfmt="{x:.2f}%",
        textcolors=("black", "white"),
        threshold=None,
        **textkw,
    ):
        if not isinstance(data, (list, np.ndarray)):
            data = im.get_array()

        if threshold is not None:
            threshold = im.norm(threshold)
        else:
            threshold = im.norm(data.max()) / 2.0

        kw = dict(horizontalalignment="center", verticalalignment="center")
        kw.update(textkw)

        if isinstance(valfmt, str):
            valfmt = ticker.StrMethodFormatter(valfmt)

        texts = []
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
                text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
                texts.append(text)
        return texts

    data = np.array(matrix, dtype=float)
    cmap = LinearSegmentedColormap.from_list("CustomColors", COLORS)

    plt.rcParams["font.family"] = "Arial"
    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    im, _ = heatmap(data, POSTURE_LABELS, POSTURE_LABELS, ax=ax, vmin=0, vmax=100, cmap=cmap)

    ax.set_yticklabels(POSTURE_LABELS, fontsize=16, fontname="Arial")
    ax.set_xticklabels(POSTURE_LABELS, fontsize=16, fontname="Arial")
    ax.set_ylabel("True Labels", fontsize=16, fontname="Arial")
    ax.set_xlabel("Predicted Labels", fontsize=16, fontname="Arial")
    ax.xaxis.set_label_position("top")

    annotate_heatmap(
        im,
        valfmt="{x:.0f}%",
        threshold=20,
        textcolors=("black", "black"),
        fontsize=16,
        fontname="Arial",
        fontweight="bold",
    )

    plt.tight_layout()
    fig.savefig(output_path, transparent=True, dpi=DPI)
    plt.close(fig)


def save_with_pil_fallback(output_path, matrix):
    from PIL import Image, ImageDraw, ImageFont

    def hex_to_rgb(value):
        value = value.lstrip("#")
        return tuple(int(value[i : i + 2], 16) for i in (0, 2, 4))

    def blend_color(value):
        value = max(0.0, min(100.0, float(value))) / 100.0
        stops = [hex_to_rgb(c) for c in COLORS]
        if value >= 1:
            return stops[-1]
        span = 1 / (len(stops) - 1)
        idx = int(value / span)
        frac = (value - idx * span) / span
        c0, c1 = stops[idx], stops[idx + 1]
        return tuple(round(c0[i] + (c1[i] - c0[i]) * frac) for i in range(3))

    def load_font(name, size):
        font_path = Path(r"C:\Windows\Fonts") / name
        if font_path.exists():
            return ImageFont.truetype(str(font_path), size=size)
        return ImageFont.load_default(size=size)

    width, height = 1880, 1620
    image = Image.new("RGBA", (width, height), (255, 255, 255, 0))
    draw = ImageDraw.Draw(image)

    font_axis = load_font("arial.ttf", 50)
    font_tick = load_font("arialbd.ttf", 50)
    font_cell = load_font("arialbd.ttf", 42)
    font_cbar = load_font("arial.ttf", 38)

    cell = 168
    grid = cell * len(POSTURE_ORDER)
    left = 380
    top = 375
    grid_width = 11

    def centered(x, y, text, font, fill=INK):
        draw.text((x, y), text, font=font, fill=fill, anchor="mm")

    centered(left + grid // 2, top - 130, "Predicted Labels", font_axis)
    for col, label in enumerate(POSTURE_LABELS):
        centered(left + col * cell + cell // 2, top - 48, label, font_tick)
    for row, label in enumerate(POSTURE_LABELS):
        centered(left - 66, top + row * cell + cell // 2, label, font_tick)

    text_box = Image.new("RGBA", (420, 100), (255, 255, 255, 0))
    d = ImageDraw.Draw(text_box)
    d.text((210, 50), "True Labels", font=font_axis, fill=INK, anchor="mm")
    rotated = text_box.rotate(90, expand=True)
    image.alpha_composite(
        rotated,
        (left - 245 - rotated.width // 2, top + grid // 2 - rotated.height // 2),
    )

    for row in range(len(POSTURE_ORDER)):
        for col in range(len(POSTURE_ORDER)):
            value = matrix[row][col]
            x1 = left + col * cell
            y1 = top + row * cell
            draw.rectangle(
                [x1, y1, x1 + cell, y1 + cell],
                fill=blend_color(value),
                outline="white",
                width=grid_width,
            )
            centered(x1 + cell // 2, y1 + cell // 2, f"{value:.0f}%", font_cell)

    cbar_x = left + grid + 90
    cbar_y = top
    cbar_w = 42
    for i in range(grid):
        value = 100 - i / max(1, grid - 1) * 100
        draw.line(
            [(cbar_x, cbar_y + i), (cbar_x + cbar_w, cbar_y + i)],
            fill=blend_color(value),
            width=1,
        )
    for tick in [0, 25, 50, 75, 100]:
        ty = cbar_y + grid - int(tick / 100 * grid)
        draw.line([(cbar_x + cbar_w + 8, ty), (cbar_x + cbar_w + 24, ty)], fill=INK, width=3)
        draw.text((cbar_x + cbar_w + 34, ty), f"{tick}%", font=font_cbar, fill=INK, anchor="lm")

    image.save(output_path, dpi=(DPI, DPI))


def save_matrix(output_path, matrix, group_title, condition_title):
    render_confusion_matrix(
        matrix,
        POSTURE_LABELS,
        f"Posture recognition: {group_title}",
        condition_title,
        output_path,
        transpose_for_display=True,
        force_column_sum_100=True,
    )
    print(f"Saved: {output_path}")


def plot_confusion_matrices():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    trials = load_residual_posture_trials()

    for group in GROUPS:
        for condition, title in CONDITIONS:
            counts, percent = build_confusion_matrix(trials, group, condition)
            print(f"{group} | {title}")
            for row in counts:
                print(row)
            print("True-label row percentages")
            for row in percent:
                print([round(v, 1) for v in row])
            save_matrix(
                OUTPUT_FILES[(group, condition)],
                percent,
                GROUPS[group]["title"],
                title,
            )


if __name__ == "__main__":
    plot_confusion_matrices()
