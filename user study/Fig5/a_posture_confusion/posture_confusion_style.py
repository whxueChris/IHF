from pathlib import Path
import math

from PIL import Image, ImageDraw, ImageFont


DPI = 600
WIDTH = 1500
HEIGHT = 1500

WHITE = (255, 255, 255, 255)
TRANSPARENT = (255, 255, 255, 0)
INK = (26, 26, 26, 255)
AXIS = (60, 60, 60, 255)
COLORS = ["#f5f8fc", "#dce8f5", "#a8c7e7", "#5b95cf", "#0b5dad"]


def _font(name, size):
    path = Path(r"C:\Windows\Fonts") / name
    if path.exists():
        return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default(size=size)


FONT_AXIS = _font("arial.ttf", 44)
FONT_TICK = _font("arial.ttf", 44)
FONT_CELL = _font("arialbd.ttf", 52)
FONT_CBAR = _font("arial.ttf", 42)


def _hex_to_rgb(value):
    value = value.lstrip("#")
    return tuple(int(value[index : index + 2], 16) for index in (0, 2, 4))


def _blend_color(value):
    fraction = max(0.0, min(100.0, float(value))) / 100.0
    colors = [_hex_to_rgb(color) for color in COLORS]
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


def _largest_remainder_percentages(values, total=100):
    """Round a percentage vector to integers that sum exactly to ``total``."""
    values = [float(value) for value in values]
    value_sum = sum(values)
    if value_sum <= 0:
        raise ValueError("Percentage vector must have a positive total")

    normalized = [value / value_sum * total for value in values]
    rounded = [math.floor(value) for value in normalized]
    remaining = total - sum(rounded)
    order = sorted(
        range(len(values)),
        key=lambda index: (normalized[index] - rounded[index], normalized[index], -index),
        reverse=True,
    )
    for index in order[:remaining]:
        rounded[index] += 1
    return rounded


def _draw_rotated_text(image, center, text, font, fill=INK):
    probe = ImageDraw.Draw(Image.new("RGBA", (1, 1)))
    box = probe.textbbox((0, 0), text, font=font)
    layer = Image.new(
        "RGBA",
        (box[2] - box[0] + 30, box[3] - box[1] + 30),
        TRANSPARENT,
    )
    layer_draw = ImageDraw.Draw(layer)
    layer_draw.text(
        (layer.width / 2, layer.height / 2),
        text,
        font=font,
        fill=fill,
        anchor="mm",
    )
    rotated = layer.rotate(90, expand=True)
    image.alpha_composite(
        rotated,
        (
            round(center[0] - rotated.width / 2),
            round(center[1] - rotated.height / 2),
        ),
    )


def render_confusion_matrix(
    matrix,
    labels,
    context,
    title,
    output_path=None,
    *,
    image_size=(WIDTH, HEIGHT),
    grid_bounds=(210.0, 200.0, 1110.0, 1100.0),
    show_colorbar=True,
    transpose_for_display=True,
    force_column_sum_100=False,
):
    """Render a compact Fig. 5A-style confusion matrix."""
    rows = len(labels)
    if rows == 0 or len(matrix) != rows or any(len(row) != rows for row in matrix):
        raise ValueError("Confusion matrix must be square and match the label count")

    image = Image.new("RGBA", image_size, TRANSPARENT)
    draw = ImageDraw.Draw(image)

    grid_left, grid_top, grid_right, grid_bottom = grid_bounds
    cell_width = (grid_right - grid_left) / rows
    cell_height = (grid_bottom - grid_top) / rows

    source_values = [[float(value) for value in row] for row in matrix]
    # Existing emoji assets use the Fig. 5A convention with true classes on
    # the x axis. Posture column-normalized matrices keep predicted classes on
    # the x axis so the visible columns have the requested interpretation.
    if transpose_for_display:
        display_values = [
            [source_values[true_index][predicted_index] for true_index in range(rows)]
            for predicted_index in range(rows)
        ]
    else:
        display_values = source_values

    # Keep raw percentages for the color map, but use a largest-remainder
    # allocation for labels when displayed columns must sum exactly to 100.
    if force_column_sum_100:
        display_labels = [[0 for _ in range(rows)] for _ in range(rows)]
        for column_index in range(rows):
            rounded_column = _largest_remainder_percentages(
                [display_values[row_index][column_index] for row_index in range(rows)]
            )
            if sum(rounded_column) != 100:
                raise AssertionError("Rounded displayed column must sum to 100%")
            for row_index, rounded_value in enumerate(rounded_column):
                display_labels[row_index][column_index] = rounded_value
    else:
        display_labels = [
            [round(value) for value in row]
            for row in display_values
        ]

    for row_index in range(rows):
        for column_index in range(rows):
            value = display_values[row_index][column_index]
            label_value = display_labels[row_index][column_index]
            x0 = grid_left + column_index * cell_width
            y0 = grid_top + row_index * cell_height
            draw.rectangle(
                (x0, y0, x0 + cell_width, y0 + cell_height),
                fill=_blend_color(value),
                outline=WHITE,
                width=3,
            )
            text_color = WHITE if value >= 55 else INK
            draw.text(
                (x0 + cell_width / 2, y0 + cell_height / 2),
                f"{label_value}%",
                font=FONT_CELL,
                fill=text_color,
                anchor="mm",
            )

    for index, label in enumerate(labels):
        draw.text(
            (grid_left + (index + 0.5) * cell_width, grid_bottom + 48),
            label,
            font=FONT_TICK,
            fill=INK,
            anchor="mm",
        )
        draw.text(
            (grid_left - 28, grid_top + (index + 0.5) * cell_height),
            label,
            font=FONT_TICK,
            fill=INK,
            anchor="rm",
        )

    if show_colorbar:
        cbar_x = grid_right + 80.0
        cbar_width = 38.0
        gradient_height = round(grid_bottom - grid_top)
        draw.text(
            (cbar_x + cbar_width / 2, grid_top - 43),
            "(%)",
            font=FONT_CBAR,
            fill=INK,
            anchor="mm",
        )
        for offset in range(gradient_height):
            value = 100.0 - offset / max(1, gradient_height - 1) * 100.0
            y = grid_top + offset
            draw.line(
                (cbar_x, y, cbar_x + cbar_width, y),
                fill=_blend_color(value),
                width=1,
            )
        for tick in (100, 75, 50, 25, 0):
            y = grid_bottom - tick / 100.0 * (grid_bottom - grid_top)
            draw.line(
                (cbar_x + cbar_width, y, cbar_x + cbar_width + 15, y),
                fill=AXIS,
                width=2,
            )
            draw.text(
                (cbar_x + cbar_width + 25, y),
                str(tick),
                font=FONT_CBAR,
                fill=INK,
                anchor="lm",
            )

    if output_path is not None:
        image.save(output_path, format="PNG", dpi=(DPI, DPI))
    return image


def save_side_by_side(left_image, right_image, output_path):
    gap = 80
    combined = Image.new(
        "RGBA",
        (left_image.width + right_image.width + gap, max(left_image.height, right_image.height)),
        TRANSPARENT,
    )
    combined.alpha_composite(left_image, (0, 0))
    combined.alpha_composite(right_image, (left_image.width + gap, 0))
    combined.save(output_path, format="PNG", dpi=(DPI, DPI))
