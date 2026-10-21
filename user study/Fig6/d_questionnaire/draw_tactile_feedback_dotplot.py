from __future__ import annotations

import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Protocol

from PIL import Image, ImageDraw, ImageFont


# Figure contract
# Core conclusion: ratings for all five dimensions are centred above the neutral
# point, with the strongest consensus for interaction comfort.
# Archetype: single-panel quantitative validation figure.
# Output: 170 mm x 65 mm, Python-only transparent PNG export plus a
# whole-figure QA composite at the supplied Fig. 5 assembly scale.

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
PARTICIPANT_CSV = DATA_DIR / "ihf_participant_info_2026-06-27T14-00-23-739Z.csv"
FEEDBACK_CSV = DATA_DIR / "tactile_feedback_questionnaire_P020_2026-06-27T13-23-51-319Z.csv"
OUT_DIR = ROOT / "output"
REFERENCE_COMPOSITE = ROOT / "reference" / "fig5_composite_before_panel_f.png"
OUT_STEM = "20260721_fig5_panel_f_journal_compact_transparent"
INTEGRATED_STEM = "20260721_fig5_integrated_panel_f_journal_qa"

DIMENSIONS = [
    ("structure_clarity", "Haptic cue legibility"),
    ("judgment_confidence", "Interpretation confidence"),
    ("perceived_helpfulness", "Perceived task usefulness"),
    ("low_cognitive_load", "Low cognitive effort"),
    ("comfort_low_burden", "Interaction comfort"),
]

GROUP_FROM_COHORT = {
    "blind_congenital": "Congenital blindness",
    "blind_acquired": "Acquired blindness",
    "clinical_low_vision": "Clinical low vision",
    "high_myopia": "High myopia",
}

GROUP_ORDER = [
    "Congenital blindness",
    "Acquired blindness",
    "Clinical low vision",
    "High myopia",
]

# Restrained, colour-blind-aware palette; marker shapes provide redundant encoding.
GROUP_STYLE = {
    "Congenital blindness": {"color": "#3B6FB6", "marker": "circle"},
    "Acquired blindness": {"color": "#D08A38", "marker": "square"},
    "Clinical low vision": {"color": "#2A8C82", "marker": "triangle"},
    "High myopia": {"color": "#8A6BBE", "marker": "diamond"},
}

TOKENS = {
    "background": "#FFFFFF",
    "ink": "#202124",
    "text": "#30343B",
    "muted": "#6B7280",
    "grid": "#E5E7EB",
    "reference": "#9CA3AF",
    "iqr": "#697386",
    "point_edge": "#FFFFFF",
}

# Full A4 text width with compact Science Advances / Nature-style typography.
# Ten drawing units equal one millimetre at final size.
W, H = 1700, 650
WIDTH_MM, HEIGHT_MM = 170.0, 65.0


PLOT_LEFT, PLOT_RIGHT = 410, 1630
PLOT_TOP, PLOT_BOTTOM = 80, 565

ROW_Y = {
    "Haptic cue legibility": 125,
    "Interpretation confidence": 225,
    "Perceived task usefulness": 325,
    "Low cognitive effort": 425,
    "Interaction comfort": 510,
}

# Marker controls. Increase/decrease these values to resize the corresponding
# visual elements; ten drawing units equal one millimetre at final size.
DOT_RADIUS = 7.0
MEDIAN_RADIUS = 9.0
IQR_LINE_WIDTH = 4.0
IQR_CAP_HALF_HEIGHT = 5.5
MIN_MARKER_GAP = 4.0
SUMMARY_OFFSET = -16.0
POINT_OFFSET = 16.0


class DrawingCanvas(Protocol):
    def text(
        self,
        x: float,
        y: float,
        value: str,
        size: float,
        fill: str,
        weight: str = "normal",
        anchor: str = "start",
    ) -> None: ...

    def rotated_text(
        self,
        x: float,
        y: float,
        value: str,
        size: float,
        fill: str,
        weight: str = "normal",
        angle: float = 90.0,
    ) -> None: ...

    def line(
        self,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        color: str,
        width: float = 1.0,
        dash: tuple[float, float] | None = None,
    ) -> None: ...

    def circle(
        self,
        x: float,
        y: float,
        radius: float,
        fill: str,
        stroke: str | None = None,
        width: float = 1.0,
    ) -> None: ...

    def polygon(
        self,
        points: list[tuple[float, float]],
        fill: str,
        stroke: str | None = None,
        width: float = 1.0,
    ) -> None: ...


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def hex_to_rgba(value: str, alpha: int = 255) -> tuple[int, int, int, int]:
    value = value.strip().lstrip("#")
    rgb = tuple(int(value[i : i + 2], 16) for i in (0, 2, 4))
    return (*rgb, alpha)


def x_for_score(score: float) -> float:
    return PLOT_LEFT + (score - 1.0) / 6.0 * (PLOT_RIGHT - PLOT_LEFT)


def percentile_linear(values: list[int], q: float) -> float:
    vals = sorted(values)
    pos = (len(vals) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return float(vals[lo])
    frac = pos - lo
    return float(vals[lo] * (1.0 - frac) + vals[hi] * frac)


def dot_offsets(n: int) -> list[tuple[float, float]]:
    """Deterministic compact beeswarm offsets for one discrete Likert score."""
    if n <= 0:
        return []
    spacing = 18.5
    if n <= 5:
        return [(0.0, (index - (n - 1) / 2.0) * spacing) for index in range(n)]

    cols = (-spacing, 0.0, spacing)
    rows = math.ceil(n / 3)
    y_values = [(index - (rows - 1) / 2.0) * spacing for index in range(rows)]
    positions = [(x, y) for y in y_values for x in cols]
    return positions[:n]


def marker_points(marker: str, x: float, y: float, radius: float) -> list[tuple[float, float]]:
    if marker == "square":
        return [
            (x - radius, y - radius),
            (x + radius, y - radius),
            (x + radius, y + radius),
            (x - radius, y + radius),
        ]
    if marker == "triangle":
        return [
            (x, y - radius * 1.15),
            (x + radius * 1.05, y + radius * 0.9),
            (x - radius * 1.05, y + radius * 0.9),
        ]
    if marker == "diamond":
        return [(x, y - radius * 1.25), (x + radius, y), (x, y + radius * 1.25), (x - radius, y)]
    raise ValueError(f"Unsupported polygon marker: {marker}")


def draw_marker(
    draw: DrawingCanvas,
    marker: str,
    x: float,
    y: float,
    radius: float,
    fill: str,
    stroke: str = TOKENS["point_edge"],
    width: float = 1.2,
) -> None:
    if marker == "circle":
        draw.circle(x, y, radius, fill, stroke, width)
    else:
        draw.polygon(marker_points(marker, x, y, radius), fill, stroke, width)


def load_data() -> list[dict[str, object]]:
    participants = read_csv(PARTICIPANT_CSV)
    participant_group: dict[str, str] = {}
    for row in participants:
        participant_id = row["participant_id"]
        cohort = row["cohort"]
        if cohort not in GROUP_FROM_COHORT:
            raise ValueError(f"Unsupported cohort for {participant_id}: {cohort}")
        participant_group[participant_id] = GROUP_FROM_COHORT[cohort]

    data: list[dict[str, object]] = []
    for row in read_csv(FEEDBACK_CSV):
        participant_id = row["participant_id"]
        if participant_id not in participant_group:
            raise ValueError(f"Missing participant information for {participant_id}")
        item: dict[str, object] = {
            "participant_id": participant_id,
            "group": participant_group[participant_id],
        }
        for column, _label in DIMENSIONS:
            item[column] = int(row[column])
        data.append(item)

    data.sort(key=lambda row: int(str(row["participant_id"])[1:]))
    return data


def build_positions(
    data: list[dict[str, object]],
) -> tuple[dict[str, dict[str, float]], list[dict[str, object]]]:
    stats: dict[str, dict[str, float]] = {}
    positions: list[dict[str, object]] = []

    for column, label in DIMENSIONS:
        scores = [int(row[column]) for row in data]
        stats[label] = {
            "q1": percentile_linear(scores, 0.25),
            "q3": percentile_linear(scores, 0.75),
            "median": percentile_linear(scores, 0.50),
            "mean": sum(scores) / len(scores),
        }

        by_score: dict[int, list[dict[str, object]]] = defaultdict(list)
        for row in data:
            by_score[int(row[column])].append(row)

        for score, rows in sorted(by_score.items()):
            for row, (dx, dy) in zip(rows, dot_offsets(len(rows))):
                positions.append(
                    {
                        "dimension": label,
                        "score": score,
                        "participant_id": row["participant_id"],
                        "group": row["group"],
                        "x": x_for_score(score) + dx,
                        "y": ROW_Y[label] + POINT_OFFSET + dy,
                    }
                )

    return stats, positions


def validate(
    data: list[dict[str, object]],
    stats: dict[str, dict[str, float]],
    positions: list[dict[str, object]],
) -> dict[str, object]:
    group_counts = Counter(str(row["group"]) for row in data)
    expected_groups = {
        "Congenital blindness": 5,
        "Acquired blindness": 2,
        "Clinical low vision": 10,
        "High myopia": 3,
    }
    assert dict(group_counts) == expected_groups, dict(group_counts)
    assert next(row for row in data if row["participant_id"] == "P015")["group"] == "Clinical low vision"

    dimension_counts: dict[str, int] = {}
    for _column, label in DIMENSIONS:
        count = sum(1 for point in positions if point["dimension"] == label)
        assert count == 20, (label, count)
        dimension_counts[label] = count

    expected_stats = {
        "Haptic cue legibility": (5.0, 4.75, 6.0, 5.10),
        "Interpretation confidence": (6.0, 5.0, 6.0, 5.55),
        "Perceived task usefulness": (5.0, 5.0, 6.0, 5.30),
        "Low cognitive effort": (6.0, 4.75, 7.0, 5.65),
        "Interaction comfort": (6.0, 6.0, 7.0, 5.90),
    }
    for label, (median, q1, q3, mean) in expected_stats.items():
        actual = stats[label]
        assert abs(actual["median"] - median) < 1e-9
        assert abs(actual["q1"] - q1) < 1e-9
        assert abs(actual["q3"] - q3) < 1e-9
        assert abs(actual["mean"] - mean) < 1e-9

    min_distance = float("inf")
    for label in ROW_Y:
        for score in range(1, 8):
            points = [
                point
                for point in positions
                if point["dimension"] == label and int(point["score"]) == score
            ]
            for i in range(len(points)):
                for j in range(i + 1, len(points)):
                    distance = math.hypot(
                        float(points[i]["x"]) - float(points[j]["x"]),
                        float(points[i]["y"]) - float(points[j]["y"]),
                    )
                    min_distance = min(min_distance, distance)
    assert min_distance >= DOT_RADIUS * 2.0 + MIN_MARKER_GAP, min_distance

    return {
        "figure_contract": {
            "core_conclusion": (
                "Central ratings are above neutral for all five dimensions, with the strongest "
                "consensus for interaction comfort."
            ),
            "archetype": "single-panel quantitative validation figure",
            "backend": "Python (Pillow)",
            "final_size_mm": [WIDTH_MM, HEIGHT_MM],
            "output_format": "transparent PNG, 600 dpi",
            "statistics_displayed": "individual responses, median, and interquartile range",
            "inferential_tests": "none; subgroup colours/shapes are descriptive only",
            "in_panel_legend": "omitted; encodings are defined in the figure caption",
            "right_side_statistics": "omitted to maximize the plotting region",
            "likert_range": "1 = strongly disagree; 7 = strongly agree",
            "final_font_sizes_pt": {
                "dimension_labels": 7.1,
                "y_axis_title": 6.8,
                "tick_labels": 6.0,
                "endpoint_labels": 5.7,
                "neutral_label": 5.1,
            },
            "panel_label": "omitted",
            "axes": {
                "x": "1-7 response scale with endpoint labels",
                "y": "Questionnaire dimension",
            },
            "font_weight": "normal",
        },
        "group_counts": dict(group_counts),
        "dimension_dot_counts": dimension_counts,
        "minimum_same_score_distance": round(min_distance, 2),
        "minimum_marker_edge_gap": round(min_distance - DOT_RADIUS * 2.0, 2),
        "stats": {
            label: {key: round(value, 2) for key, value in values.items()}
            for label, values in stats.items()
        },
    }


class PillowCanvas:
    def __init__(self, pixel_width: int):
        self.scale = pixel_width / W
        self.pixel_width = pixel_width
        self.pixel_height = round(H * self.scale)
        self.image = Image.new("RGBA", (self.pixel_width, self.pixel_height), (255, 255, 255, 0))
        self.draw = ImageDraw.Draw(self.image)
        self._font_cache: dict[tuple[int, bool], ImageFont.FreeTypeFont] = {}

    def _font(self, size: float, bold: bool) -> ImageFont.FreeTypeFont:
        pixel_size = max(1, round(size * self.scale))
        key = (pixel_size, bold)
        if key not in self._font_cache:
            font_name = "arialbd.ttf" if bold else "arial.ttf"
            font_path = Path("C:/Windows/Fonts") / font_name
            if font_path.exists():
                self._font_cache[key] = ImageFont.truetype(str(font_path), size=pixel_size)
            else:
                fallback = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
                try:
                    self._font_cache[key] = ImageFont.truetype(fallback, size=pixel_size)
                except OSError:
                    mac_font = Path('/System/Library/Fonts/Supplemental') / ('Arial Bold.ttf' if bold else 'Arial.ttf')
                    self._font_cache[key] = ImageFont.truetype(str(mac_font), size=pixel_size) if mac_font.exists() else ImageFont.load_default(size=pixel_size)
        return self._font_cache[key]

    def _xy(self, x: float, y: float) -> tuple[float, float]:
        return x * self.scale, y * self.scale

    def text(
        self,
        x: float,
        y: float,
        value: str,
        size: float,
        fill: str,
        weight: str = "normal",
        anchor: str = "start",
    ) -> None:
        anchors = {"start": "lm", "middle": "mm", "end": "rm"}
        self.draw.text(
            self._xy(x, y),
            value,
            font=self._font(size, weight in {"bold", "600", "700"}),
            fill=hex_to_rgba(fill),
            anchor=anchors[anchor],
        )

    def rotated_text(
        self,
        x: float,
        y: float,
        value: str,
        size: float,
        fill: str,
        weight: str = "normal",
        angle: float = 90.0,
    ) -> None:
        font = self._font(size, weight in {"bold", "600", "700"})
        bbox = font.getbbox(value)
        padding = max(4, round(4 * self.scale))
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        layer = Image.new(
            "RGBA",
            (text_width + padding * 2, text_height + padding * 2),
            (255, 255, 255, 0),
        )
        layer_draw = ImageDraw.Draw(layer)
        layer_draw.text(
            (padding - bbox[0], padding - bbox[1]),
            value,
            font=font,
            fill=hex_to_rgba(fill),
        )
        rotated = layer.rotate(angle, expand=True, resample=Image.Resampling.BICUBIC)
        cx, cy = self._xy(x, y)
        self.image.alpha_composite(
            rotated,
            (round(cx - rotated.width / 2), round(cy - rotated.height / 2)),
        )

    def line(
        self,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        color: str,
        width: float = 1.0,
        dash: tuple[float, float] | None = None,
    ) -> None:
        line_width = max(1, round(width * self.scale))
        if not dash:
            self.draw.line((*self._xy(x1, y1), *self._xy(x2, y2)), fill=hex_to_rgba(color), width=line_width)
            return

        length = math.hypot(x2 - x1, y2 - y1)
        if length == 0:
            return
        dx, dy = (x2 - x1) / length, (y2 - y1) / length
        cursor = 0.0
        while cursor < length:
            segment_end = min(length, cursor + dash[0])
            sx1, sy1 = x1 + dx * cursor, y1 + dy * cursor
            sx2, sy2 = x1 + dx * segment_end, y1 + dy * segment_end
            self.draw.line(
                (*self._xy(sx1, sy1), *self._xy(sx2, sy2)),
                fill=hex_to_rgba(color),
                width=line_width,
            )
            cursor += dash[0] + dash[1]

    def circle(
        self,
        x: float,
        y: float,
        radius: float,
        fill: str,
        stroke: str | None = None,
        width: float = 1.0,
    ) -> None:
        cx, cy = self._xy(x, y)
        r = radius * self.scale
        self.draw.ellipse(
            (cx - r, cy - r, cx + r, cy + r),
            fill=hex_to_rgba(fill),
            outline=hex_to_rgba(stroke) if stroke else None,
            width=max(1, round(width * self.scale)),
        )

    def polygon(
        self,
        points: list[tuple[float, float]],
        fill: str,
        stroke: str | None = None,
        width: float = 1.0,
    ) -> None:
        scaled = [self._xy(x, y) for x, y in points]
        self.draw.polygon(scaled, fill=hex_to_rgba(fill))
        if stroke:
            self.draw.line(
                [*scaled, scaled[0]],
                fill=hex_to_rgba(stroke),
                width=max(1, round(width * self.scale)),
                joint="curve",
            )

    def save(self, path: Path, dpi: int, format_name: str) -> None:
        self.image.save(path, format=format_name, dpi=(dpi, dpi))


def draw_figure(
    draw: DrawingCanvas,
    stats: dict[str, dict[str, float]],
    positions: list[dict[str, object]],
    _group_counts: Counter[str],
) -> None:
    # Group encodings and summary definitions belong in the figure caption,
    # leaving the panel itself focused on the response distribution.
    neutral_x = x_for_score(4)
    draw.line(neutral_x, PLOT_TOP, neutral_x, PLOT_BOTTOM, TOKENS["reference"], 1.2, (6.0, 6.0))
    draw.text(neutral_x, 55, "Neutral", 18, TOKENS["muted"], anchor="middle")

    # Summary intervals sit behind the participant-level observations.
    for label, row_y in ROW_Y.items():
        summary_y = row_y + SUMMARY_OFFSET
        values = stats[label]
        q1_x = x_for_score(values["q1"])
        q3_x = x_for_score(values["q3"])
        median_x = x_for_score(values["median"])

        draw.text(PLOT_LEFT - 28, row_y + 2, label, 25, TOKENS["ink"], anchor="end")
        draw.line(q1_x, summary_y, q3_x, summary_y, TOKENS["iqr"], IQR_LINE_WIDTH)
        draw.line(
            q1_x,
            summary_y - IQR_CAP_HALF_HEIGHT,
            q1_x,
            summary_y + IQR_CAP_HALF_HEIGHT,
            TOKENS["iqr"],
            1.6,
        )
        draw.line(
            q3_x,
            summary_y - IQR_CAP_HALF_HEIGHT,
            q3_x,
            summary_y + IQR_CAP_HALF_HEIGHT,
            TOKENS["iqr"],
            1.6,
        )
        draw_marker(
            draw,
            "diamond",
            median_x,
            summary_y,
            MEDIAN_RADIUS,
            TOKENS["ink"],
            TOKENS["ink"],
            0.9,
        )

    # Individual observations are the primary evidence.
    for point in positions:
        group = str(point["group"])
        style = GROUP_STYLE[group]
        draw_marker(
            draw,
            str(style["marker"]),
            float(point["x"]),
            float(point["y"]),
            DOT_RADIUS,
            str(style["color"]),
            width=0.9,
        )

    # Minimal journal-style axis treatment. The endpoint definitions carry the
    # Likert range without a separate explanatory legend.
    draw.line(PLOT_LEFT, PLOT_TOP, PLOT_LEFT, PLOT_BOTTOM, TOKENS["ink"], 1.8)
    draw.line(PLOT_LEFT, PLOT_BOTTOM, PLOT_RIGHT, PLOT_BOTTOM, TOKENS["ink"], 1.8)

    for row_y in ROW_Y.values():
        draw.line(PLOT_LEFT - 7, row_y, PLOT_LEFT, row_y, TOKENS["ink"], 1.3)

    for score in range(1, 8):
        x = x_for_score(score)
        draw.line(x, PLOT_BOTTOM, x, PLOT_BOTTOM + 6, TOKENS["ink"], 1.0)
        draw.text(x, PLOT_BOTTOM + 27, str(score), 21, TOKENS["ink"], anchor="middle")

    draw.rotated_text(
        32,
        (PLOT_TOP + PLOT_BOTTOM) / 2.0,
        "Questionnaire dimension",
        24,
        TOKENS["ink"],
        angle=90.0,
    )

    draw.text(PLOT_LEFT, 625, "Strongly disagree", 20, TOKENS["muted"])
    draw.text(PLOT_RIGHT, 625, "Strongly agree", 20, TOKENS["muted"], anchor="end")


def draw_dashed_line(
    draw: ImageDraw.ImageDraw,
    start: tuple[float, float],
    end: tuple[float, float],
    fill: tuple[int, int, int, int],
    width: int,
    dash: float = 14.0,
    gap: float = 11.0,
) -> None:
    x1, y1 = start
    x2, y2 = end
    length = math.hypot(x2 - x1, y2 - y1)
    if length == 0:
        return
    ux, uy = (x2 - x1) / length, (y2 - y1) / length
    cursor = 0.0
    while cursor < length:
        segment_end = min(length, cursor + dash)
        draw.line(
            (
                x1 + ux * cursor,
                y1 + uy * cursor,
                x1 + ux * segment_end,
                y1 + uy * segment_end,
            ),
            fill=fill,
            width=width,
        )
        cursor += dash + gap


def draw_dashed_rounded_border(
    image: Image.Image,
    box: tuple[int, int, int, int],
    radius: int = 42,
) -> None:
    x0, y0, x1, y1 = box
    draw = ImageDraw.Draw(image)
    color = (96, 111, 138, 255)
    width = 4

    draw_dashed_line(draw, (x0 + radius, y0), (x1 - radius, y0), color, width)
    draw_dashed_line(draw, (x1, y0 + radius), (x1, y1 - radius), color, width)
    draw_dashed_line(draw, (x1 - radius, y1), (x0 + radius, y1), color, width)
    draw_dashed_line(draw, (x0, y1 - radius), (x0, y0 + radius), color, width)

    arc_specs = [
        ((x0, y0, x0 + radius * 2, y0 + radius * 2), 180, 270),
        ((x1 - radius * 2, y0, x1, y0 + radius * 2), 270, 360),
        ((x1 - radius * 2, y1 - radius * 2, x1, y1), 0, 90),
        ((x0, y1 - radius * 2, x0 + radius * 2, y1), 90, 180),
    ]
    for arc_box, start_angle, end_angle in arc_specs:
        angle = start_angle
        while angle < end_angle:
            draw.arc(
                arc_box,
                start=angle,
                end=min(angle + 8, end_angle),
                fill=color,
                width=width,
            )
            angle += 14


def save_integrated_qa(panel_path: Path) -> dict[str, object]:
    if not REFERENCE_COMPOSITE.exists():
        return {"available": False, "reason": "whole-figure reference not found"}

    reference = Image.open(REFERENCE_COMPOSITE).convert("RGBA")
    panel = Image.open(panel_path).convert("RGBA")

    # Preserve the completed upper panels, replace the old panel F, and shorten
    # the complete canvas to expose any scale or whitespace mismatch immediately.
    panel_top = 2199
    panel_x = 18
    panel_y = panel_top + 5
    panel_width = reference.width - panel_x * 2
    panel_height = round(panel.height * panel_width / panel.width)
    border_bottom = panel_y + panel_height + 5

    assembled = Image.new("RGBA", (reference.width, border_bottom + 5), (0, 0, 0, 0))
    assembled.alpha_composite(reference.crop((0, 0, reference.width, panel_top)), (0, 0))
    resized_panel = panel.resize((panel_width, panel_height), Image.Resampling.LANCZOS)
    assembled.alpha_composite(resized_panel, (panel_x, panel_y))
    draw_dashed_rounded_border(assembled, (5, panel_top, reference.width - 6, border_bottom))

    transparent_path = OUT_DIR / f"{INTEGRATED_STEM}_transparent.png"
    assembled.save(transparent_path, format="PNG", dpi=reference.info.get("dpi", (150, 150)))

    white_preview = Image.new("RGBA", assembled.size, (255, 255, 255, 255))
    white_preview.alpha_composite(assembled)
    white_path = OUT_DIR / f"{INTEGRATED_STEM}_white_preview.png"
    white_preview.convert("RGB").save(
        white_path,
        format="PNG",
        dpi=reference.info.get("dpi", (150, 150)),
    )

    return {
        "available": True,
        "transparent_composite": str(transparent_path),
        "white_background_preview": str(white_path),
        "pixel_size": list(assembled.size),
        "panel_box": [panel_x, panel_y, panel_x + panel_width, panel_y + panel_height],
    }


def save_outputs(
    stats: dict[str, dict[str, float]],
    positions: list[dict[str, object]],
    group_counts: Counter[str],
) -> dict[str, object]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    base = OUT_DIR / OUT_STEM

    png_path = base.with_suffix(".png")
    png_width = round(WIDTH_MM / 25.4 * 600)
    png = PillowCanvas(png_width)
    draw_figure(png, stats, positions, group_counts)
    png.save(png_path, dpi=600, format_name="PNG")

    alpha = png.image.getchannel("A")
    alpha_min, alpha_max = alpha.getextrema()
    assert alpha_min == 0 and alpha_max == 255, (alpha_min, alpha_max)
    assert all(png.image.getpixel(corner)[3] == 0 for corner in [(0, 0), (png.pixel_width - 1, 0), (0, png.pixel_height - 1), (png.pixel_width - 1, png.pixel_height - 1)])

    integrated_qa = save_integrated_qa(png_path)

    return {
        "png": str(png_path),
        "mode": png.image.mode,
        "pixel_size": [png.pixel_width, png.pixel_height],
        "dpi": 600,
        "transparent_background": True,
        "nontransparent_bbox": list(alpha.getbbox() or ()),
        "integrated_qa": integrated_qa,
    }


def main() -> None:
    data = load_data()
    stats, positions = build_positions(data)
    validation = validate(data, stats, positions)
    outputs = save_outputs(stats, positions, Counter(str(row["group"]) for row in data))
    validation["export_checks"] = outputs

    validation_path = OUT_DIR / f"{OUT_STEM}_validation.json"
    validation_path.write_text(json.dumps(validation, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps({"outputs": outputs, "validation": str(validation_path)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
