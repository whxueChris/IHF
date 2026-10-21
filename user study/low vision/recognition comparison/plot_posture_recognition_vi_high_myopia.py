from pathlib import Path
import csv
from collections import defaultdict
from math import comb

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT
OUTPUT_PNG = OUTPUT_DIR / "posture_recognition_residual_visual_access_combined.png"
DPI = 600

INPUT_FILES = [
    (
        "corrected_p001_p003_p004",
        ROOT / "P004_2026-06-24T12-03-06_trial_summary_posture_corrected.csv",
    ),
    (
        "corrected_p002_p011",
        ROOT / "P011_2026-06-25T10-44-32_trial_summary_posture_corrected.csv",
    ),
    (
        "supplemented_p012_p019",
        ROOT / "P019_2026-06-26T22-29-20_trial_summary_posture_supplemented.csv",
    ),
    (
        "supplemented_p020",
        ROOT / "P020_2026-06-27T21-22-19_trial_summary_posture_supplemented.csv",
    ),
]

COHORT_BY_TABLE = {
    "P001": "Clinical low vision",
    "P004": "Clinical low vision",
    "P008": "Clinical low vision",
    "P009": "Clinical low vision",
    "P010": "Clinical low vision",
    "P011": "Clinical low vision",
    "P013": "Clinical low vision",
    "P015": "Clinical low vision",
    "P016": "Clinical low vision",
    "P017": "Clinical low vision",
    "P018": "High myopia",
    "P019": "High myopia",
    "P020": "High myopia",
}

BLUE = "#1565c0"
ORANGE = "#ff6d00"
GREY_LINE = "#d9d9d9"
AXIS = "#333333"
INK = "#111111"


def load_font(name, size):
    font_path = Path(r"C:\Windows\Fonts") / name
    if font_path.exists():
        return ImageFont.truetype(str(font_path), size=size)
    return ImageFont.load_default(size=size)


FONT_TITLE = load_font("arialbd.ttf", 58)
FONT_SUBTITLE = load_font("arialbd.ttf", 46)
FONT_AXIS = load_font("arial.ttf", 42)
FONT_TICK = load_font("arial.ttf", 36)
FONT_LABEL = load_font("arialbd.ttf", 42)
FONT_SMALL = load_font("arial.ttf", 34)


def draw_centered(draw, xy, text, font, fill=INK):
    draw.text(xy, text, font=font, fill=fill, anchor="mm")


def draw_rotated_text(base, text, xy, font, fill=INK, angle=90):
    box = Image.new("RGBA", (500, 120), (255, 255, 255, 0))
    d = ImageDraw.Draw(box)
    d.text((250, 60), text, font=font, fill=fill, anchor="mm")
    rotated = box.rotate(angle, expand=True)
    base.alpha_composite(rotated, (xy[0] - rotated.width // 2, xy[1] - rotated.height // 2))


def load_posture_accuracy():
    stats = defaultdict(lambda: {"valid": 0, "correct": 0})
    for label, path in INPUT_FILES:
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
                pid = row["participant_id"]
                if pid not in COHORT_BY_TABLE:
                    continue
                if row["condition"] not in {"visual_only", "visual_ihf"}:
                    continue
                if row.get("skipped", "").strip().lower() == "true":
                    continue
                key = (pid, row["condition"])
                stats[key]["valid"] += 1
                if row.get("correct", "").strip().lower() == "true":
                    stats[key]["correct"] += 1

    rows = []
    for pid, cohort in COHORT_BY_TABLE.items():
        visual = stats.get((pid, "visual_only"))
        ihf = stats.get((pid, "visual_ihf"))
        if not visual or not ihf or visual["valid"] == 0 or ihf["valid"] == 0:
            continue
        rows.append(
            {
                "participant_id": pid,
                "cohort": cohort,
                "visual_only": visual["correct"] / visual["valid"] * 100,
                "visual_ihf": ihf["correct"] / ihf["valid"] * 100,
            }
        )
    return rows


def sign_test_label(values):
    diffs = [row["visual_ihf"] - row["visual_only"] for row in values]
    improved = sum(1 for d in diffs if d > 1e-9)
    not_improved = len(diffs) - improved
    if improved == 0:
        return "n.s."
    tail = sum(comb(len(diffs), k) for k in range(min(improved, not_improved) + 1))
    p_value = min(1.0, 2 * tail / (2 ** len(diffs)))
    if p_value < 0.001:
        return "P < 0.001"
    if p_value < 0.01:
        return "P < 0.01"
    if p_value < 0.05:
        return "P < 0.05"
    return f"P = {p_value:.2f}"


def draw_marker(draw, center, color, shape, radius=16):
    x, y = center
    if shape == "triangle":
        points = [(x, y - radius), (x - radius, y + radius), (x + radius, y + radius)]
        draw.polygon(points, fill=color)
        draw.line(points + [points[0]], fill="white", width=3, joint="curve")
        return
    draw.ellipse(
        [x - radius, y - radius, x + radius, y + radius],
        fill=color,
        outline="white",
        width=3,
    )


def marker_shape(row):
    if row["cohort"] == "Clinical low vision":
        return "triangle"
    return "circle"


def draw_panel(image, draw, x0, y0, width, height, title, values):
    plot_left = x0 + 170
    plot_right = x0 + width - 120
    plot_top = y0 + 250
    plot_bottom = y0 + height - 165
    plot_center = (plot_left + plot_right) // 2
    x_gap = min(520, int((plot_right - plot_left) * 0.45))
    x_visual = plot_center - x_gap // 2
    x_ihf = plot_center + x_gap // 2

    def y_from_accuracy(value):
        return plot_bottom - (value / 100) * (plot_bottom - plot_top)

    def jitter_offsets(condition):
        groups = defaultdict(list)
        for row in values:
            groups[round(row[condition], 1)].append(row["participant_id"])

        offsets = {}
        for participant_ids in groups.values():
            participant_ids = sorted(participant_ids)
            count = len(participant_ids)
            if count == 1:
                local_offsets = [0]
            else:
                span = min(180, 24 * (count - 1))
                local_offsets = [
                    -span / 2 + (span * idx / (count - 1)) for idx in range(count)
                ]
            for participant_id, offset in zip(participant_ids, local_offsets):
                offsets[participant_id] = offset
        return offsets

    draw_centered(draw, (x0 + width // 2, y0 + 46), title, FONT_SUBTITLE)
    draw_centered(draw, (x0 + width // 2, y0 + 102), f"(n = {len(values)} participants)", FONT_SMALL)

    # Axes
    draw.line([(plot_left, plot_top), (plot_left, plot_bottom)], fill=AXIS, width=4)
    draw.line([(plot_left, plot_bottom), (plot_right, plot_bottom)], fill=AXIS, width=4)
    for tick in [0, 25, 50, 75, 100]:
        y = y_from_accuracy(tick)
        draw.line([(plot_left - 16, y), (plot_left, y)], fill=AXIS, width=4)
        draw.text((plot_left - 28, y), str(tick), font=FONT_TICK, fill=INK, anchor="rm")

    draw_rotated_text(image, "Accuracy (%)", (x0 + 46, (plot_top + plot_bottom) // 2), FONT_AXIS)
    draw_centered(draw, (x_visual, plot_bottom + 76), "Visual-only", FONT_TICK)
    draw_centered(draw, (x_ihf, plot_bottom + 76), "Visual + IHF", FONT_TICK)

    visual_offsets = jitter_offsets("visual_only")
    ihf_offsets = jitter_offsets("visual_ihf")

    # Participant trajectories
    for row in values:
        x_visual_jittered = x_visual + visual_offsets[row["participant_id"]]
        x_ihf_jittered = x_ihf + ihf_offsets[row["participant_id"]]
        y_visual = y_from_accuracy(row["visual_only"])
        y_ihf = y_from_accuracy(row["visual_ihf"])
        draw.line(
            [(x_visual_jittered, y_visual), (x_ihf_jittered, y_ihf)],
            fill=GREY_LINE,
            width=4,
        )

    radius = 17
    for row in values:
        x_visual_jittered = x_visual + visual_offsets[row["participant_id"]]
        x_ihf_jittered = x_ihf + ihf_offsets[row["participant_id"]]
        y_visual = y_from_accuracy(row["visual_only"])
        y_ihf = y_from_accuracy(row["visual_ihf"])
        shape = marker_shape(row)
        draw_marker(draw, (x_visual_jittered, y_visual), BLUE, shape, radius)
        draw_marker(draw, (x_ihf_jittered, y_ihf), ORANGE, shape, radius)

    mean_visual = sum(row["visual_only"] for row in values) / len(values)
    mean_ihf = sum(row["visual_ihf"] for row in values) / len(values)
    y_mean_visual = y_from_accuracy(mean_visual)
    y_mean_ihf = y_from_accuracy(mean_ihf)
    draw.line([(x_visual - 48, y_mean_visual), (x_visual + 48, y_mean_visual)], fill=BLUE, width=5)
    draw.line([(x_ihf - 48, y_mean_ihf), (x_ihf + 48, y_mean_ihf)], fill=ORANGE, width=5)
    draw.text((x_visual - 132, y_mean_visual - 4), f"{mean_visual:.1f}%", font=FONT_LABEL, fill=BLUE, anchor="rm")
    draw.text((x_ihf + 132, y_mean_ihf - 4), f"{mean_ihf:.1f}%", font=FONT_LABEL, fill=ORANGE, anchor="lm")

    # Significance bracket, Fig. 5D-style.
    bracket_y = plot_top - 50
    draw.line([(x_visual, bracket_y), (x_ihf, bracket_y)], fill=AXIS, width=3)
    draw.line([(x_visual, bracket_y), (x_visual, bracket_y + 20)], fill=AXIS, width=3)
    draw.line([(x_ihf, bracket_y), (x_ihf, bracket_y + 20)], fill=AXIS, width=3)
    draw_centered(draw, ((x_visual + x_ihf) // 2, bracket_y - 32), sign_test_label(values), FONT_SMALL)


def draw_legend(draw, center_x, y):
    row1_y = y
    row2_y = y + 80
    item1_x = center_x - 520
    item2_x = center_x - 120
    item3_x = center_x + 310

    draw_marker(draw, (item1_x, row1_y), BLUE, "circle", 19)
    draw.text((item1_x + 42, row1_y), "Visual-only", font=FONT_SMALL, fill=INK, anchor="lm")
    draw_marker(draw, (item2_x, row1_y), ORANGE, "circle", 19)
    draw.text((item2_x + 42, row1_y), "Visual + IHF", font=FONT_SMALL, fill=INK, anchor="lm")
    draw.line([(item3_x, row1_y), (item3_x + 90, row1_y)], fill=GREY_LINE, width=5)
    draw.text((item3_x + 112, row1_y), "Same participant", font=FONT_SMALL, fill=INK, anchor="lm")

    item4_x = center_x - 300
    item5_x = center_x + 200
    draw_marker(draw, (item4_x, row2_y), "#555555", "triangle", 17)
    draw.text((item4_x + 42, row2_y), "Clinical low vision", font=FONT_SMALL, fill=INK, anchor="lm")
    draw_marker(draw, (item5_x, row2_y), "#555555", "circle", 17)
    draw.text((item5_x + 42, row2_y), "High myopia", font=FONT_SMALL, fill=INK, anchor="lm")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = load_posture_accuracy()
    clinical = [row for row in rows if row["cohort"] == "Clinical low vision"]
    high_myopia = [row for row in rows if row["cohort"] == "High myopia"]

    for label, values in [("Clinical low vision", clinical), ("High myopia", high_myopia)]:
        print(label)
        for row in values:
            print(row["participant_id"], round(row["visual_only"], 1), round(row["visual_ihf"], 1))

    combined = clinical + high_myopia

    image = Image.new("RGBA", (1800, 1540), (255, 255, 255, 255))
    draw = ImageDraw.Draw(image)

    draw_centered(
        draw,
        (900, 95),
        "Posture recognition under residual visual access",
        FONT_TITLE,
    )
    draw_panel(
        image,
        draw,
        120,
        175,
        1560,
        1060,
        "Clinical low vision + high myopia",
        combined,
    )
    draw_legend(draw, 900, 1300)
    draw.text(
        (150, 1475),
        "Statistics at participant level. Accuracy is computed as correct / valid posture trials.",
        font=FONT_SMALL,
        fill=INK,
        anchor="lm",
    )

    image.save(OUTPUT_PNG, dpi=(DPI, DPI))
    print(f"Saved: {OUTPUT_PNG}")


if __name__ == "__main__":
    main()
