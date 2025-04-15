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
:root {
    --background-light: #f5f7fa;
    --background-dark: #1e1e1e;
    --text-light: #2c3e50;
    --text-dark: #ecf0f1;
    --card-bg-light: #ffffff;
    --card-bg-dark: #2c3e50;
    --primary-color: #3498db;
    --secondary-color: #e74c3c;
    --hover-color: #2980b9;
    --border-color-light: #dfe6e9;
    --border-color-dark: #7f8c8d;
}

body {
    background: var(--background-light);
    color: var(--text-light);
    font: normal 600 1.3em/1.5 'Poppins', sans-serif;
    margin: 0;
    padding: 20px;
    transition: background 0.3s ease, color 0.3s ease;
}

body.dark {
    background: var(--background-dark);
    color: var(--text-dark);
}

/* Header */
#header {
    text-align: center;
    margin-bottom: 30px;
}

#header h1 {
    font-size: 2.5em;
    color: var(--text-light);
    margin-bottom: 10px;
}

#header h2 {
    font-size: 1.5em;
    color: var(--text-light);
}

body.dark #header h1,
body.dark #header h2 {
    color: var(--text-dark);
}

/* Main Layout */
#main-row {
    display: flex;
    gap: 20px;
}

#model-settings-column {
    flex: 1;
    max-width: 300px;
}

#evaluation-column {
    flex: 2;
}

/* Model Settings */
#model-settings-group {
    background: var(--card-bg-light);
    border-radius: 12px;
    padding: 20px;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    margin-bottom: 30px;
    transition: background 0.3s ease, box-shadow 0.3s ease;
}

body.dark #model-settings-group {
    background: var(--card-bg-dark);
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
}

#model-settings-title {
    font-size: 1.5em;
    color: var(--text-light);
    margin-bottom: 20px;
}

body.dark #model-settings-title {
    color: var(--text-dark);
}

.dropdown .gr-dropdown {
    border-radius: 8px;
    border: 1px solid var(--border-color-light);
    padding: 10px;
    background: var(--card-bg-light);
    transition: border-color 0.3s ease, box-shadow 0.3s ease;
}

body.dark .dropdown .gr-dropdown {
    background: var(--card-bg-dark);
    border-color: var(--border-color-dark);
}

.dropdown .gr-dropdown:hover {
    border-color: var(--primary-color);
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
}

/* Buttons */
.primary-button {
    background-color: var(--primary-color);
    color: #ffffff;
    border-radius: 8px;
    padding: 10px 20px;
    transition: background-color 0.3s ease, transform 0.3s ease;
}

.primary-button:hover {
    background-color: var(--hover-color);
    transform: translateY(-2px);
}

.secondary-button {
    background-color: var(--secondary-color);
    color: #ffffff;
    border-radius: 8px;
    padding: 10px 20px;
    transition: background-color 0.3s ease, transform 0.3s ease;
}

.secondary-button:hover {
    background-color: #c0392b;
    transform: translateY(-2px);
}

/* Textbox */
.textbox {
    margin-bottom: 20px;
}

.textbox .gr-textbox {
    border-radius: 8px;
    border: 1px solid var(--border-color-light);
    padding: 10px;
    background: var(--card-bg-light);
    transition: border-color 0.3s ease, box-shadow 0.3s ease;
}

body.dark .textbox .gr-textbox {
    background: var(--card-bg-dark);
    border-color: var(--border-color-dark);
}

.textbox .gr-textbox:hover {
    border-color: var(--primary-color);
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
}

/* Details Section */
.details-section {
    margin-top: 30px;
    padding: 20px;
    background: var(--card-bg-light);
    border-radius: 12px;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    border: 1px solid var(--border-color-light);
    transition: all 0.3s ease; /* 平滑过渡 */
    overflow: hidden; /* 隐藏溢出内容 */
}

.details-section[style*="display: none"] {
    height: 0 !important;
    margin: 0 !important;
    padding: 0 !important;
}

body.dark .details-section {
    background: var(--card-bg-dark);
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
    border-color: var(--border-color-dark);
}

.details-section h3 {
    font: normal 600 1.3em/1.5 'Poppins', sans-serif;
    margin-bottom: 15px;
    color: var(--text-light);
    text-transform: uppercase;
}

body.dark .details-section h3 {
    color: var(--text-dark);
}

/* Preformatted Text */
.details-section pre {
    background: var(--card-bg-light);
    padding: 12px;
    border-radius: 8px;
    font-family: 'Fira Code', monospace;
    color: var(--text-light);
    line-height: 1.5;
    white-space: pre-wrap;
    word-wrap: break-word;
    overflow: hidden;
}

body.dark .details-section pre {
    background: var(--card-bg-dark);
    color: var(--text-dark);
}

.details-section p {
    margin-top: 15px;
    font-size: 1.1em;
    color: var(--text-light);
    font-weight: bold;
}

body.dark .details-section p {
    color: var(--text-dark);
}

/* Calibration Details */
.calibration-details {
    font: normal 600 1.3em/1.5 'Poppins', sans-serif;
    margin-top: 15px;
    padding-left: 20px;
    list-style-type: disc;
}

.calibration-details li {
    margin-bottom: 10px;
    color: var(--text-light);  /* 默认模式下的文字颜色 */
}

body.dark .calibration-details li {
    color: var(--text-dark);  /* 暗黑模式下的文字颜色 */
}

.calibration-details li b {
    color: var(--primary-color);  /* 默认模式下的加粗文字颜色 */
}

body.dark .calibration-details li b {
    color: var(--hover-color);  /* 暗黑模式下的加粗文字颜色 */
}

.calibration-details p {
    margin-top: 15px;
    font-size: 1.1em;
    color: var(--text-light);  /* 默认模式下的文字颜色 */
    font-weight: bold;
}

body.dark .calibration-details p {
    color: var(--text-dark);  /* 暗黑模式下的文字颜色 */
}
"""