from __future__ import annotations
import os
import sys
import gc

import gradio as gr
from gradio.components import Dropdown
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
import requests
import pandas as pd
import json

# 添加到 sys.path 以导入其他模块
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config import FINETUNED_JUDGE_MODELS, PROPRIETARY_MODELS
from utils import (
    show_batch_calibration_mode,
    show_calibration_mode,
    update_batch_calibration_mode,
    update_calibration_mode,
    update_model_options,
    load_model_based_on_type,
    manual_evaluate,
    enable_evaluate_button,
    batch_evaluation
)
from webui.theme import Seafoam, css
from webui.evaluation import evaluate, evaluate_batch, toggle_details, calibrated_evaluation, calibrated_evaluation_batch, evaluate_batch_with_api
from models.model import load_model, clear_model
with gr.Blocks(theme=Seafoam(), css=css) as demo:
    gr.Markdown(
        """
        # 🌟 LLM Evaluation Web Application
        ### 基于 LLM-as-a-Judge 的大模型评估网页应用
        """
    )
    
    state = gr.State({"llm": None, "tokenizer": None, "sampling_params": None, "model_name": None})

    with gr.Row():
        with gr.Column(scale=1):
            with gr.Group(elem_classes="panel"):
                gr.Markdown("### 📋 模型设置")

                model_type_selector = gr.Radio(
                    label="选择模型类型",
                    choices=["微调裁判模型", "专有模型"],
                    value="微调裁判模型",
                    interactive=True  # 确保可交互
                )

                model_selector = gr.Dropdown(
                    label="选择模型",
                    choices=list(FINETUNED_JUDGE_MODELS.keys()),
                    value=list(FINETUNED_JUDGE_MODELS.keys())[0],
                    interactive=True  # 确保可交互
                )
                
                with gr.Row():
                    load_model_btn = gr.Button("加载模型", variant="primary")
                    unload_model_btn = gr.Button("卸载模型", variant="secondary")
                
                model_load_output = gr.Textbox(
                    label="模型状态",
                    interactive=False
                )


    # 绑定模型类型选择器的变更事件
    model_type_selector.change(
        fn=update_model_choices,
        inputs=[model_type_selector],
        outputs=[model_selector]
    )

    # 绑定加载模型按钮的点击事件
    load_model_btn.click(
        fn=load_model_based_on_type,
        inputs=[model_selector, model_type_selector, state],
        outputs=[model_load_output, load_model_btn]
    )

    # 绑定卸载模型按钮的点击事件
    unload_model_btn.click(
        fn=clear_model,
        inputs=[state],
        outputs=[model_load_output]
    )

    with gr.Tabs() as tabs:
        with gr.TabItem("📝 手动评估", elem_classes="tab-content"):
            with gr.Row():
                with gr.Column(scale=1):
                    instruction_input = gr.Textbox(
                        label="指令",
                        placeholder="请输入评估指令……",
                        lines=3,
                        elem_classes="input-box"
                    )
                
            with gr.Row():
                with gr.Column(scale=1):
                    answer1_input = gr.Textbox(
                        label="助手 1 回答",
                        placeholder="请输入第一个回答……",
                        lines=3,
                        elem_classes="input-box"
                    )
                with gr.Column(scale=1):
                    answer2_input = gr.Textbox(
                        label="助手 2 回答", 
                        placeholder="请输入第二个回答……",
                        lines=3,
                        elem_classes="input-box"
                    )
            
            with gr.Row():
                evaluation_mode_selector = gr.Radio(
                    choices=["Direct Evaluation", "Chain of Thought (CoT)"],
                    label="评估模式",
                    value="Direct Evaluation",
                    elem_classes="eval-mode"
                )
                calibration_mode = gr.Checkbox(
                    label="启用校准模式 (仅支持专有模型)",
                    value=False,
                    visible=False,
                    elem_classes="calibration-toggle"
                )
            
            evaluate_btn = gr.Button(
                "开始评估",
                interactive=False,
                elem_classes="primary evaluate-btn"
            )
            
            with gr.Group(elem_classes="results-panel"):
                result_output = gr.Textbox(
                    label="评估结果",
                    interactive=False,
                    elem_classes="result-box"
                )
                details_output = gr.HTML(
                    visible=False,
                    elem_classes="details-section"
                )
                details_button = gr.Button(
                    "显示详情",
                    elem_classes="secondary details-btn"
                )

        with gr.TabItem("📊 批量评估", elem_classes="tab-content"):
            with gr.Group(elem_classes="panel"):
                file_input = gr.File(
                    label="上传数据文件 (CSV/JSON)",
                    elem_classes="file-input"
                )
                save_path_input = gr.Textbox(
                    label="保存路径",
                    placeholder="请输入结果保存路径……",
                    elem_classes="input-box"
                )
                
                with gr.Row():
                    batch_mode_selector = gr.Radio(
                        choices=["Direct Evaluation", "Chain of Thought (CoT)"],
                        label="批量评估模式",
                        value="Direct Evaluation",
                        elem_classes="eval-mode"
                    )
                    batch_calibration_mode = gr.Checkbox(
                        label="启用批量校准模式 (仅支持专有模型)",
                        value=False,
                        visible=False,
                        elem_classes="calibration-toggle"
                    )
                
                batch_evaluate_btn = gr.Button(
                    "开始批量评估",
                    interactive=False,
                    elem_classes="primary evaluate-btn"
                )
                
                with gr.Group(elem_classes="results-panel"):
                    batch_result_output = gr.Textbox(
                        label="批量评估结果",
                        interactive=False,
                        elem_classes="result-box"
                    )
                    gr.Markdown(
                        """
                        #### 📋 支持的文件格式
                        - CSV文件: 包含 instruction, answer1, answer2 列
                        - JSON文件: 包含相应字段的数组
                        """,
                        elem_classes="format-help"
                    )

    # 添加页脚
    gr.Markdown(
        """
        <div class="footer">
            <p>Designed by Hongli Zhou</p>
            <p>Version 1.0.0 | <a href="https://github.com/Joe-Hall-Lee/LLMEvalWeb" target="_blank">GitHub</a></p>
        </div>
        """,
        elem_classes="footer"
    )

if __name__ == "__main__":
    demo.launch(pwa=True)