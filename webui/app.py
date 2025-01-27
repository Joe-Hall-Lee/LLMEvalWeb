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
    update_model_choices,
    load_model_based_on_type,
    manual_evaluate,
    enable_evaluate_button,
    batch_evaluation,
    update_model_type,
    update_eval_mode
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
    
    state = gr.State({
        "tokenizer": None,
        "model_type": "微调裁判模型",  # 默认模型类型
        "eval_mode": "单模型评估"  # 默认评估模式
    })

    with gr.Row():
        with gr.Column(scale=1):
            with gr.Group():
                gr.Markdown("### 📋 模型设置")

                # 添加评估模式选择器
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

                # 添加专有模型选择器，默认隐藏
                proprietary_model_selector = gr.Dropdown(
                    label="选择专有模型",
                    choices=list(PROPRIETARY_MODELS.keys()),
                    value=list(PROPRIETARY_MODELS.keys())[0],
                    interactive=True,
                    visible=False,
                    elem_classes=["dropdown"]
                )

                # 添加阈值输入框，使用 Slider 组件
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
            
            # 将评估模式选择器和校准按钮放在同一列，校准按钮在下方
            with gr.Column():
                # 评估模式选择器
                evaluation_mode_selector = gr.Radio(
                    choices=["直接评估", "思维链"],
                    label="推理策略",
                    value="直接评估",
                    visible=False  # 默认隐藏
                )

                # 校准模式开关
                calibration_mode = gr.Checkbox(
                    label="启用校准",
                    value=False,
                    visible=False  # 默认隐藏
                )

            # 当模型类型改变时更新评估模式选择器和校准开关的可见性
            model_type_selector.change(
                fn=lambda model_type: [
                    gr.update(visible=model_type == "专有模型"),  # 如果是专有模型，显示评估模式
                    gr.update(visible=model_type == "专有模型"),  # 如果是专有模型，显示校准开关
                ],
                inputs=[model_type_selector],
                outputs=[evaluation_mode_selector, calibration_mode]
            )
            model_type_selector.change(
                fn=update_model_type,
                inputs=[model_type_selector, state],
                outputs=[model_selector]
            )

            evaluate_btn = gr.Button(
                "开始评估",
                interactive=False
            )

            model_load_output.change(enable_evaluate_button, inputs=model_load_output, outputs=evaluate_btn)
            model_load_output.change(lambda x: gr.update(interactive=True), inputs=model_load_output, outputs=load_model_btn)
            
            with gr.Group():
                result_output = gr.Textbox(
                    label="评估结果",
                    interactive=False
                )
                details_output = gr.HTML("<div class='details-section'><h3>详情将在这里显示</h3></div>", visible=False, elem_classes=["details-section"])
                details_button = gr.Button("显示详情")
                evaluate_btn.click(manual_evaluate, inputs=[instruction_input, answer1_input, answer2_input, evaluation_mode_selector, state, calibration_mode], outputs=[result_output, details_output])
                details_button.click(toggle_details, outputs=[details_output, details_button])

        with gr.TabItem("📊 批量评估"):

            file_input = gr.File(
                label="上传数据文件 (CSV/JSON)"
            )
            save_path_input = gr.Textbox(
                label="保存路径",
                placeholder="请输入结果保存路径……"
            )
                
            # 批量评估模式和校准按钮的容器
            with gr.Column():
                batch_mode_selector = gr.Radio(
                    choices=["直接评估", "思维链"],
                    label="评估模式",
                    value="直接评估",
                    visible=False
                )
                batch_calibration_mode = gr.Checkbox(
                    label="启用校准",
                    value=False,
                    visible=False
                )
            
            # 将开始批量评估按钮单独放置，与模式和校准组件分隔开
            batch_evaluate_btn = gr.Button(
                "开始批量评估",
                interactive=False
            )
            
            with gr.Group():
                batch_result_output = gr.Textbox(
                    label="批量评估结果",
                    interactive=False
                )
                gr.Markdown(
                    """
                    #### 📋 支持的文件格式
                    - CSV 文件: 包含 instruction, answer1, answer2 列
                    - JSON 文件: 包含相应字段的数组
                    """
                )

        # 绑定模型类型选择器的变更事件
        model_type_selector.change(
            fn=lambda model_type: [
                gr.update(visible=model_type == "专有模型"),
                gr.update(visible=model_type == "专有模型"),
            ],
            inputs=[model_type_selector],
            outputs=[batch_mode_selector, batch_calibration_mode]
        )

        batch_evaluate_btn.click(batch_evaluation, inputs=[file_input, save_path_input, batch_mode_selector, state, batch_calibration_mode], outputs=batch_result_output)
        model_load_output.change(enable_evaluate_button, inputs=model_load_output, outputs=batch_evaluate_btn)

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

    
    # 绑定评估模式选择器的变更事件
    eval_mode_selector.change(
        fn=update_eval_mode,
        inputs=[eval_mode_selector, state],
        outputs=[model_type_selector, proprietary_model_selector, threshold_input, model_type_selector, evaluation_mode_selector, calibration_mode]
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
        inputs=[model_selector, proprietary_model_selector, eval_mode_selector, state],
        outputs=[model_load_output, load_model_btn]
    )

    # 绑定卸载模型按钮的点击事件
    unload_model_btn.click(
        fn=clear_model,
        inputs=[state],
        outputs=[model_load_output]
    )

if __name__ == "__main__":
    demo.launch(pwa=True)