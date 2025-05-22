from __future__ import annotations
from typing import Iterable
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
    --background-light: #f9fafb;
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

/* Prevent scrollbar hiding when modal is open */
body.modal-open {
    overflow: auto !important;
    padding-right: 0 !important;
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
    background: linear-gradient(135deg, #e6f0fa 0%, #d4f4e8 100%); /* 添加渐变背景 */
    border-radius: 16px; /* 更现代的圆角 */
    padding: 24px; /* 增加内边距，显得更宽松 */
    box-shadow: 0 6px 12px rgba(0, 0, 0, 0.15); /* 更柔和的阴影 */
    margin-bottom: 30px;
    transition: background 0.3s ease, box-shadow 0.3s ease, transform 0.3s ease; /* 添加 transform 过渡 */
}

#model-settings-group:hover {
    transform: translateY(-4px); /* 鼠标悬停时轻微上移，增加交互感 */
    box-shadow: 0 8px 16px rgba(0, 0, 0, 0.2); /* 悬停时阴影加深 */
}

body.dark #model-settings-group {
    background: linear-gradient(135deg, #2c3e50 0%, #1a3c34 100%); /* 深色模式渐变 */
    box-shadow: 0 6px 12px rgba(0, 0, 0, 0.3);
}

body.dark #model-settings-group:hover {
    transform: translateY(-4px);
    box-shadow: 0 8px 16px rgba(0, 0, 0, 0.4);
}

#model-settings-title {
    font-size: 1.6em; /* 标题稍大 */
    font-weight: 700; /* 加粗 */
    color: var(--text-light);
    margin-bottom: 24px; /* 增加标题下间距 */
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
    border-radius: 6px;
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
    border-radius: 6px;
    padding: 10px 20px;
    transition: background-color 0.3s ease, transform 0.3s ease;
}

.secondary-button:hover {
    background-color: #c0392b;
    transform: translateY(-2px);
}

/* Details Button (显示详情按钮) */
.details-button {
    display: inline-flex; /* 使用 flex 布局，确保内容居中 */
    align-items: center; /* 垂直居中 */
    justify-content: center; /* 水平居中 */
    border-radius: 6px;
    text-align: center;
    transition: background-color 0.3s ease, transform 0.3s ease;
}

.details-button:hover {
    background-color: var(--hover-color);
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
    transition: all 0.3s ease;
    overflow: hidden;
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
    color: #2d3748;
    text-transform: uppercase;
}

body.dark .details-section h3 {
    color: #d1d5db;
}

.details-section pre {
    background: var(--card-bg-light);
    padding: 12px;
    border-radius: 8px;
    font-family: 'Fira Code', monospace;
    color: #2d3748;
    line-height: 1.5;
    white-space: pre-wrap;
    word-wrap: break-word;
    overflow-y: auto;
}

body.dark .details-section pre {
    background: var(--card-bg-dark);
    color: #d1d5db;
}

.details-section p {
    margin-top: 15px;
    font-size: 1.1em;
    color: #2d3748;
    font-weight: bold;
}

body.dark .details-section p {
    color: #d1d5db;
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
    color: #2d3748;
}

body.dark .calibration-details li {
    color: #d1d5db;
}

.calibration-details li b {
    color: var(--primary-color);
}

body.dark .calibration-details li b {
    color: var(--hover-color);
}

.calibration-details p {
    margin-top: 15px;
    font-size: 1.1em;
    color: #2d3748;
    font-weight: bold;
}

body.dark .calibration-details p {
    color: #d1d5db;
}

/* Modal Overlay (遮罩层) */
.modal-overlay {
    position: fixed;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background: rgba(0, 0, 0, 0.5);
    z-index: 999;
    animation: fadeInOverlay 0.3s ease-out;
}

/* Add modal-open class to body when modal is visible */
.modal-overlay[style*="display: block"] ~ * {
    /* Ensure the body gets the modal-open class when modal-overlay is visible */
}
body:has(.modal-overlay[style*="display: block"]) {
    overflow: auto !important;
    padding-right: 0 !important;
}

/* Modal Container (弹窗样式) */
.modal-container {
    position: fixed;
    top: 10%;
    left: 15%;
    right: 15%;
    bottom: 10%;
    background: #ffffff;
    border-radius: 10px;
    border: 2px solid #e5e7eb;
    box-shadow: 0 6px 16px rgba(0, 0, 0, 0.1);
    z-index: 1001;
    padding: 20px;
    overflow: hidden;
    animation: fadeIn 0.3s ease-out;
}

body.dark .modal-container {
    background: #1f252a;
    border: 2px solid #374151;
}

.modal-container .close-button {
    position: absolute;
    top: 12px;
    right: 12px;
    width: 32px;
    height: 32px;
    background: #ef4444;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    transition: background 0.2s, transform 0.2s;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    z-index: 1009;
}

.modal-container .close-button:hover {
    background: #dc2626;
    transform: scale(1.05);
}

.modal-container .close-button::before {
    content: '×';
    color: white;
    font-size: 20px;
    font-weight: bold;
}

.modal-content {
    overflow-y: auto;
    padding: 16px;
    background: #ffffff;
}

body.dark .modal-content {
    background: #1f252a;
}

.modal-content h3 {
    color: #1f252a;
    margin-bottom: 12px;
    font-size: 18px;
    font-weight: 700;
    border-bottom: 2px solid #e5e7eb;
    padding-bottom: 4px;
}

body.dark .modal-content h3 {
    color: #e2e8f0;
    border-bottom: 2px solid #374151;
}

.modal-content hr {
    border: none;
    border-top: 1px solid #e5e7eb;
    margin: 12px 0;
}

body.dark .modal-content hr {
    border-top: 1px solid #374151;
}

.modal-content pre {
    background: #f9fafb;
    border: 1px solid #e5e7eb;
    padding: 12px;
    border-radius: 6px;
    font-size: 15px;
    line-height: 1.6;
    overflow-x: auto;
    color: #1f252a;
}

body.dark .modal-content pre {
    background: #2d3748;
    border: 1px solid #374151;
    color: #e2e8f0;
}

.pretty-scroll::-webkit-scrollbar {
    width: 6px;
}

.pretty-scroll::-webkit-scrollbar-track {
    background: transparent;
}

.pretty-scroll::-webkit-scrollbar-thumb {
    background: #e5e7eb;
    border-radius: 3px;
}

.pretty-scroll::-webkit-scrollbar-thumb:hover {
    background: #b0b7c0;
}

body.dark .pretty-scroll::-webkit-scrollbar-thumb {
    background: #4b5563;
}

body.dark .pretty-scroll::-webkit-scrollbar-thumb:hover {
    background: #6b7280;
}

@keyframes fadeIn {
    from { opacity: 0; transform: translateY(8px); }
    to { opacity: 1; transform: translateY(0); }
}

@keyframes fadeInOverlay {
    from { opacity: 0; }
    to { opacity: 1; }
}

/* 统计摘要样式 */
.stats-summary {
    background: var(--card-bg-light);
    border-radius: 12px;
    padding: 20px;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    margin-bottom: 20px;
}

body.dark .stats-summary {
    background: var(--card-bg-dark);
}

.stats-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
    gap: 15px;
}

.stat-item {
    text-align: center;
    padding: 15px;
    background: rgba(255, 255, 255, 0.1);
    border-radius: 8px;
}

.stat-value {
    font-size: 24px;
    font-weight: 700;
    color: var(--primary-color);
}

.stat-label {
    font-size: 14px;
    color: var(--text-light);
    margin-top: 8px;
}

body.dark .stat-label {
    color: var(--text-dark);
}

/* 可视化标签页专属样式 */
#visualization_tab {
    padding: 20px;
    background: var(--card-bg-light);
    border-radius: 12px;
}

#visualization_tab .plot-container {
    margin-bottom: 20px;
    border: 1px solid var(--border-color-light);
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
}

body.dark #visualization_tab {
    background: var(--card-bg-dark);
}

/* 报告选择器 */
#report_selector {
    max-width: 300px;
    margin-left: auto;
}

/* 统计面板增强 */
.stats-summary {
    padding: 15px;
    background: rgba(255,255,255,0.9);
    backdrop-filter: blur(5px);
}
"""

