from __future__ import annotations
import os
import sys
import gc
import gradio as gr
from gradio.components import Dropdown
from transformers import AutoTokenizer
import requests
import pandas as pd
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config import FINETUNED_JUDGE_MODELS, PROPRIETARY_MODELS
from utils import (
    update_batch_calibration_mode, clear_model,
    update_calibration_mode, update_model_choices, load_model_based_on_type,
    manual_evaluate, enable_evaluate_button, batch_evaluation, update_model_type, update_eval_mode
)
from helpers import (
    show_batch_calibration_mode, show_calibration_mode
)
from webui.theme import Seafoam, css
from visualization import analyze_results, update_report_list

# 在 gr.Tabs 外部定义可视化组件
stats_html = gr.HTML(visible=True)
comp_plot = gr.Plot(label="模型对比分析", visible=True)

with gr.Blocks(theme=Seafoam(), css=css) as demo:
    gr.Markdown(
        """
        # 🌟 LLM Evaluation Web Application
        ### 基于 LLM-as-a-Judge 的大模型评估网页应用
        """
    )
    
    state = gr.State({
        "tokenizer": None,
        "model_type": "微调裁判模型",
        "eval_mode": "单模型评估",
        "confidence_threshold": 0.5,
        "finetuned_model_name": list(FINETUNED_JUDGE_MODELS.keys())[0],
        "proprietary_model_name": list(PROPRIETARY_MODELS.keys())[0]
    })

    with gr.Row():
        with gr.Column(scale=1):
            with gr.Group():
                gr.Markdown("### 📋 模型设置")
                eval_mode_selector = gr.Radio(
                    label="评估模式",
                    choices=["单模型评估", "级联评估"],
                    value="单模型评估",
                    interactive=True,
                    elem_classes=["radio-group"]
                )
                model_type_selector = gr.Radio(
                    label="选择模型类型",
                    choices=["微调裁判模型", "专有模型"],
                    value="微调裁判模型",
                    interactive=True,
                    elem_classes=["radio-group"]
                )
                model_selector = gr.Dropdown(
                    label="选择微调裁判模型",
                    choices=list(FINETUNED_JUDGE_MODELS.keys()),
                    value=list(FINETUNED_JUDGE_MODELS.keys())[0],
                    interactive=True,
                    elem_classes=["dropdown"]
                )
                proprietary_model_selector = gr.Dropdown(
                    label="选择专有模型",
                    choices=list(PROPRIETARY_MODELS.keys()),
                    value=list(PROPRIETARY_MODELS.keys())[0],
                    interactive=True,
                    visible=False,
                    elem_classes=["dropdown"]
                )
                threshold_input = gr.Slider(
                    label="置信度阈值",
                    value=0.5,
                    minimum=0.0,
                    maximum=1.0,
                    step=0.01,
                    interactive=True,
                    visible=False,
                    elem_classes=["slider"]
                )
                with gr.Row():
                    load_model_btn = gr.Button("加载模型", variant="primary", elem_classes=["primary-button"])
                    unload_model_btn = gr.Button("卸载模型", variant="secondary", elem_classes=["secondary-button"])
                model_load_output = gr.Textbox(
                    label="模型状态",
                    interactive=False,
                    elem_classes=["textbox"]
                )

    with gr.Tabs() as tabs:
        with gr.TabItem("📝 手动评估"):
            with gr.Row():
                with gr.Column(scale=1):
                    instruction_input = gr.Textbox(
                        label="指令",
                        placeholder="请输入评估指令……",
                        lines=3
                    )
            with gr.Row():
                with gr.Column(scale=1):
                    answer1_input = gr.Textbox(
                        label="大模型 1 回答",
                        placeholder="请输入第一个回答……",
                        lines=3
                    )
                with gr.Column(scale=1):
                    answer2_input = gr.Textbox(
                        label="大模型 2 回答", 
                        placeholder="请输入第二个回答……",
                        lines=3
                    )
            with gr.Column():
                evaluation_mode_selector = gr.Radio(
                    choices=["直接评估", "思维链"],
                    label="推理策略",
                    value="直接评估",
                    visible=False
                )
                calibration_mode = gr.Checkbox(
                    label="启用校准",
                    value=False,
                    visible=False
                )
            model_type_selector.change(
                fn=show_calibration_mode,
                inputs=[model_type_selector, eval_mode_selector],
                outputs=[evaluation_mode_selector, calibration_mode]
            )
            model_type_selector.change(
                fn=update_model_type,
                inputs=[model_type_selector, state],
                outputs=[state]
            ).then(
                fn=update_model_choices,
                inputs=[model_type_selector],
                outputs=[model_selector, proprietary_model_selector]
            ).then(
                fn=update_calibration_mode,
                inputs=[model_type_selector],
                outputs=[state]
            )
            evaluate_btn = gr.Button("开始评估", interactive=False)
            model_load_output.change(enable_evaluate_button, inputs=model_load_output, outputs=evaluate_btn)

            with gr.Group():
                result_output = gr.Textbox(label="评估结果", interactive=False)
                details_button = gr.Button("显示详情", elem_classes=["details-button"])

                # 遮罩层和弹窗
                modal_overlay = gr.HTML('<div class="modal-overlay" style="display: none;"></div>', visible=False)
                details_panel = gr.Group(visible=False, elem_classes=["modal-container"])
                with details_panel:
                    close_button = gr.Button("", elem_classes=["close-button"])
                    details_output = gr.HTML(label="评估详情", elem_classes=["modal-content", "pretty-scroll"])

            def show_details(verdict, details):
                return verdict, details, gr.update(visible=True), gr.update(visible=True)

            def hide_details():
                return gr.update(visible=False), gr.update(visible=False)

            evaluate_btn.click(
                fn=manual_evaluate,
                inputs=[instruction_input, answer1_input, answer2_input, evaluation_mode_selector, state, calibration_mode],
                outputs=[result_output, details_output]
            )
            details_button.click(
                fn=lambda verdict, details: show_details(verdict, details),
                inputs=[result_output, details_output],
                outputs=[result_output, details_output, modal_overlay, details_panel]
            )
            close_button.click(
                fn=hide_details,
                outputs=[modal_overlay, details_panel]
            )

        with gr.TabItem("📊 批量评估"):
            file_input = gr.File(label="上传数据文件 (CSV/JSON)")
            with gr.Column():
                batch_mode_selector = gr.Radio(
                    choices=["直接评估", "思维链"],
                    label="推理策略",
                    value="直接评估",
                    visible=False
                )
                batch_calibration_mode = gr.Checkbox(label="启用校准", value=False, visible=False)
            batch_evaluate_btn = gr.Button("开始批量评估", interactive=False)
            batch_result_output = gr.Textbox(label="批量评估结果", interactive=False)
            report_download = gr.File(label="评估报告下载", visible=False, interactive=False) 
            gr.Markdown(
                """
                #### 📋 支持的文件格式
                - CSV 文件: 包含 instruction, answer1, answer2 列
                - JSON 文件: 包含相应字段的数组
                """
            )

            eval_mode_selector.change(
                fn=update_eval_mode,
                inputs=[eval_mode_selector, state],
                outputs=[
                    model_type_selector, proprietary_model_selector, threshold_input,
                    model_type_selector, evaluation_mode_selector, calibration_mode,
                    batch_mode_selector, batch_calibration_mode
                ]
            )

            model_type_selector.change(
                fn=show_batch_calibration_mode,
                inputs=[model_type_selector, eval_mode_selector],
                outputs=[batch_mode_selector, batch_calibration_mode]
            ).then(
                fn=update_batch_calibration_mode,
                inputs=[model_type_selector],
                outputs=[state]
            )

            batch_evaluate_btn.click(
                fn=batch_evaluation,
                inputs=[file_input, batch_mode_selector, state, batch_calibration_mode],
                outputs=[batch_result_output, report_download]
            ).then(
                fn=lambda: gr.update(visible=True),
                outputs=report_download
            ).then(
                fn=analyze_results,
                inputs=report_download,
                outputs=[stats_html, comp_plot]
            )
            model_load_output.change(enable_evaluate_button, inputs=model_load_output, outputs=batch_evaluate_btn)
        with gr.TabItem("📈 结果可视化", id="visualization_tab"):
            with gr.Row():
                with gr.Column(scale=1):
                    report_selector = gr.Dropdown(
                        label="选择评估报告",
                        interactive=True,
                        visible=True
                    )
                    refresh_btn = gr.Button("🔄 刷新报告列表", size="sm")
                                            
            with gr.Row():
                with gr.Column(scale=1):
                    stats_html = gr.HTML(label="统计摘要", visible=True, elem_id="stats_html")
                with gr.Column(scale=2):
                    comp_plot = gr.Plot(label="模型对比分析", visible=True, elem_id="comp_plot")

            # 刷新报告列表
            refresh_btn.click(
                fn=update_report_list,
                outputs=report_selector
            )

            # 选择报告后分析结果
            report_selector.change(
                fn=analyze_results,
                inputs=report_selector,
                outputs=[stats_html, comp_plot]
            )
    gr.Markdown(
        """
        <div class="footer">
            <p>Designed by Hongli Zhou</p>
        </div>
        """,
        elem_classes="footer"
    )

    model_type_selector.change(
        fn=lambda model_type: [
            gr.update(visible=model_type == "微调裁判模型"),
            gr.update(visible=model_type == "专有模型"),
        ],
        inputs=[model_type_selector],
        outputs=[model_selector, proprietary_model_selector]
    )

    model_selector.change(
        fn=lambda selected_model, s: {**s, "finetuned_model_name": selected_model},
        inputs=[model_selector, state],
        outputs=[state]
    )
    proprietary_model_selector.change(
        fn=lambda prop_model, s: {**s, "proprietary_model_name": prop_model},
        inputs=[proprietary_model_selector, state],
        outputs=[state]
    )

    load_model_btn.click(
        fn=load_model_based_on_type,
        inputs=[model_type_selector, model_selector, proprietary_model_selector, eval_mode_selector, state],
        outputs=[model_load_output, load_model_btn]
    )

    unload_model_btn.click(
        fn=clear_model,
        inputs=[state],
        outputs=[model_load_output]
    )

    threshold_input.change(
        fn=lambda threshold, s: {**s, "confidence_threshold": threshold},
        inputs=[threshold_input, state],
        outputs=state
    )

if __name__ == "__main__":
    demo.launch()