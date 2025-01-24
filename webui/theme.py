from __future__ import annotations
from typing import Iterable
import gradio as gr
from gradio.themes.base import Base
from gradio.themes.utils import colors, fonts, sizes

class Seafoam(Base):
    def __init__(
        self,
        *,
        primary_hue: colors.Color | str = colors.emerald,
        secondary_hue: colors.Color | str = colors.blue,
        neutral_hue: colors.Color | str = colors.blue,
        spacing_size: sizes.Size | str = sizes.spacing_md,
        radius_size: sizes.Size | str = sizes.radius_md,
        text_size: sizes.Size | str = sizes.text_lg,
        font: fonts.Font
        | str
        | Iterable[fonts.Font | str] = (
            fonts.GoogleFont("Quicksand"),
            "ui-sans-serif",
            "sans-serif",
        ),
        font_mono: fonts.Font
        | str
        | Iterable[fonts.Font | str] = (
            fonts.GoogleFont("IBM Plex Mono"),
            "ui-monospace",
            "monospace",
        ),
    ):
        super().__init__(
            primary_hue="orange",
            secondary_hue="blue",
            neutral_hue="sky",
            spacing_size=spacing_size,
            radius_size=radius_size,
            text_size=text_size,
            font=font,
            font_mono=font_mono,
        )

        super().set(
            # Colors
            slider_color="*neutral_900",
            slider_color_dark="*neutral_500",
            accordion_text_color="*body_text_color",
            accordion_text_color_dark="*body_text_color",
            table_text_color="*body_text_color",
            table_text_color_dark="*body_text_color",
            body_text_color="*neutral_900",
            block_label_text_color="*body_text_color",
            block_title_text_color="*body_text_color",
            body_text_color_subdued="*neutral_700",
            body_background_fill="repeating-linear-gradient(45deg, *secondary_200, *secondary_200 10px, *secondary_50 10px, *secondary_50 20px)",
            body_background_fill_dark="repeating-linear-gradient(45deg, *secondary_800, *secondary_800 10px, *secondary_900 10px, *secondary_900 20px)",
            # Buttons
            button_border_width="2px",
            button_primary_border_color="*neutral_900",
            button_primary_background_fill="*neutral_900",
            button_primary_background_fill_hover="*neutral_700",
            button_primary_text_color="white",
            button_primary_background_fill_dark="*neutral_600",
            button_primary_background_fill_hover_dark="*neutral_600",
            button_primary_text_color_dark="white",
            button_secondary_border_color="*neutral_900",
            button_secondary_background_fill="white",
            button_secondary_background_fill_hover="white",
            button_secondary_background_fill_dark="*neutral_700",
            button_secondary_border_color_hover="*neutral_400",
            button_secondary_text_color="*neutral_900",
            button_secondary_text_color_hover="*neutral_400",
            button_cancel_border_color="*neutral_900",
            button_cancel_background_fill="*button_secondary_background_fill",
            button_cancel_background_fill_hover="*button_secondary_background_fill_hover",
            button_cancel_text_color="*button_secondary_text_color",

            checkbox_label_border_color="*checkbox_background_color",
            checkbox_label_border_color_hover="*button_secondary_border_color_hover",
            checkbox_label_border_color_selected="*button_primary_border_color",
            checkbox_label_border_width="*button_border_width",
            checkbox_background_color="*input_background_fill",
            checkbox_label_background_fill_selected="*button_primary_background_fill",
            checkbox_label_text_color_selected="*button_primary_text_color",
            # Padding
            checkbox_label_padding="*spacing_sm",
            button_large_padding="*spacing_lg",
            button_small_padding="*spacing_sm",
            # Borders
            shadow_drop_lg="0 1px 4px 0 rgb(0 0 0 / 0.1)",
            block_shadow="none",
            block_shadow_dark="*shadow_drop_lg",
            # Block Labels
            block_title_text_weight="600",
            block_label_text_weight="600",
            block_label_text_size="*text_md",
        )


# Custom CSS for the application
css = """
/* General styles */
body {
    background-color: var(--body-background-fill);
    color: var(--neutral-900);
    font-family: 'Poppins', sans-serif;
}


/* Details Section */
.details-section {
    margin-top: 30px;
    padding: 20px;
    background-color: var(--neutral-100);
    border-radius: 12px;
    box-shadow: 0 6px 20px rgba(0, 0, 0, 0.2);
    border: 1px solid var(--neutral-300);
}

.details-section h3 {
    margin-bottom: 15px;
    font-size: 1.3em;
    color: var(--neutral-900);
    text-transform: uppercase;
}

/* Calibration Details List */
.calibration-details {
    list-style-type: none;
    padding: 12px;
    margin: 0;
    background-color: var(--neutral-50);
    border-radius: 8px;
    box-shadow: 0 4px 10px rgba(0, 0, 0, 0.1);
}

.calibration-details li {
    margin-bottom: 10px;
    font-size: 1em;
    color: var(--neutral-700);
}

.calibration-details li b {
    color: var(--neutral-900);
}

/* Preformatted Text */
.details-section pre {
    background-color: var(--neutral-200);
    padding: 12px;
    border-radius: 8px;
    font-family: 'Fira Code', monospace;
    color: var(--neutral-900);
    line-height: 1.5;
    white-space: pre-wrap;  /* 自动换行 */
    word-wrap: break-word; /* 强制单词换行 */
    overflow: hidden;      /* 去掉滚动条 */
}

/* Additional Styling for Verdict */
.details-section p {
    margin-top: 15px;
    font-size: 1.1em;
    color: var(--neutral-800);
    font-weight: bold;
}
"""
