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

from webui.theme import Seafoam, css
from webui.evaluation import evaluate, evaluate_batch, toggle_details, calibrated_evaluation, calibrated_evaluation_batch, evaluate_batch_with_api
from models.model import load_model, clear_model

# 模型选项
open_source_models = [
    ("Qwen2.5-3B-Instruct", "models/Qwen2.5-3B-Instruct"),
    ("Vicuna-7B-v1.5", "models/vicuna-7b-v1.5")
]

proprietary_models = [
    ("Qwen-Plus", "qwen-plus")
]

# Gradio 界面部分
with gr.Blocks(theme=Seafoam(), css=css) as demo:
    gr.Markdown("# LLM Evaluation Web Application")
    gr.Markdown("### A web application for evaluating the outputs of large language models.")

    state = gr.State({"llm": None, "tokenizer": None, "sampling_params": None, "model_name": None, "model_type": "Fine-tuned Judge Model"})

    model_type_selector = gr.Radio(label="Select Model Type", choices=["Fine-tuned Judge Model", "Proprietary Model"], value="Fine-tuned Judge Model")
    model_selector = gr.Dropdown(label="Select Model", choices=open_source_models, value="Qwen2.5-3B-Instruct")
    model_load_output = gr.Textbox(label="Model Load Status", interactive=False)
    load_model_btn = gr.Button("Load Model")
    unload_model_btn = gr.Button("Unload Model")

    # 启用评估按钮
    def enable_evaluate_button(load_status):
        if "successfully" in load_status.lower():  # 判断加载状态是否成功
            return gr.update(interactive=True)
        return gr.update(interactive=False)


    # 根据选择的模型类型动态更新模型选项
    def update_model_options(model_type):
        if model_type == "Fine-tuned Judge Model":
            return gr.update(choices=open_source_models, value=open_source_models[0][0])
        else:
            return gr.update(choices=proprietary_models, value=proprietary_models[0][0])

    model_type_selector.change(update_model_options, inputs=model_type_selector, outputs=model_selector)

    def load_model_based_on_type(model_name, model_type, state):
        state["model_type"] = model_type
        if model_type == "Fine-tuned Judge Model":
            if model_name.startswith("models/"):
                load_status, button_state = load_model(model_name, state)
                return load_status, gr.update(interactive=True)  # 始终保持按钮可用
        elif model_type == "Proprietary Model":
            if not model_name.startswith("models/"):
                state["model_name"] = model_name
                return f"API model {model_name} loaded successfully", gr.update(interactive=True)
        return "无效的模型类型或路径", gr.update(interactive=True)  # 保证交互状态



    load_model_btn.click(load_model_based_on_type, inputs=[model_selector, model_type_selector, state], outputs=[model_load_output, load_model_btn])

    unload_model_btn.click(clear_model, inputs=[state], outputs=model_load_output)

    # Manual Input Tab
    with gr.Tab("Manual Input"):
        instruction_input = gr.Textbox(label="Instruction", placeholder="Enter the instruction here...", lines=3)
        answer1_input = gr.Textbox(label="Assistant 1", placeholder="Enter the first answer here...", lines=3)
        answer2_input = gr.Textbox(label="Assistant 2", placeholder="Enter the second answer here...", lines=3)
        evaluation_mode_selector = gr.Radio(choices=["Direct Evaluation", "Chain of Thought (CoT)"], label="Evaluation Mode", value="Direct Evaluation")
        calibration_mode = gr.Checkbox(label="Enable Calibration (Proprietary Models Only)", value=False, visible=False) # 初始隐藏

        # 手动评估中的校准模式显示逻辑
        def update_calibration_mode(model_type):
            if model_type == "Proprietary Model":
                return gr.update(visible=True, value=False)  # 显示校准模式并重置为 False
            return gr.update(visible=False, value=False)  # 隐藏校准模式并重置为 False


        def show_calibration_mode(model_type):
            return gr.update(visible=model_type == "Proprietary Model")

        model_type_selector.change(update_calibration_mode, inputs=model_type_selector, outputs=calibration_mode)


        result_output = gr.Textbox(label="Evaluation Summary", interactive=False)
        details_output = gr.HTML("<div class='details-section'><h3>Details will appear here</h3></div>", visible=False, elem_classes=["details-section"])
        details_button = gr.Button("Show Details")
        evaluate_btn = gr.Button("Evaluate", interactive=False)

        def manual_evaluate(instruction, answer1, answer2, mode, state, calibration_mode):
            model_type = state.get("model_type")
            if model_type == "Proprietary Model":
                model_name = state.get("model_name")
                if not model_name:
                    return "请先加载 API 模型", "", gr.update(visible=False)
                if calibration_mode:
                    return calibrated_evaluation(instruction, answer1, answer2, mode, model_name=model_name)
                return evaluate(instruction, answer1, answer2, mode, model_name=model_name)

            llm = state.get("llm")
            tokenizer = state.get("tokenizer")
            sampling_params = state.get("sampling_params")
            if llm is None or tokenizer is None or sampling_params is None:
                return "请先加载模型", "", gr.update(visible=False)

            if calibration_mode:
                return "校准模式只能用于专有模型。", "", gr.update(visible=False)

            return evaluate(instruction, answer1, answer2, mode, llm=llm, tokenizer=tokenizer, sampling_params=sampling_params)

        evaluate_btn.click(manual_evaluate, inputs=[instruction_input, answer1_input, answer2_input, evaluation_mode_selector, state, calibration_mode], outputs=[result_output, details_output]) #inputs 添加 evaluation_mode_selector
        details_button.click(toggle_details, outputs=[details_output, details_button])

        model_load_output.change(enable_evaluate_button, inputs=model_load_output, outputs=evaluate_btn)
        model_load_output.change(lambda x: gr.update(interactive=True), inputs=model_load_output, outputs=load_model_btn)


    # Batch Evaluation Tab
    with gr.Tab("Batch Evaluation"):
        file_input = gr.File(label="Upload CSV or JSON File")
        save_path_input = gr.Textbox(label="Save Path", placeholder="Enter output file path")
        batch_mode_selector = gr.Radio(choices=["Direct Evaluation", "Chain of Thought (CoT)"], label="Batch Evaluation Mode", value="Direct Evaluation")
        batch_calibration_mode = gr.Checkbox(label="Enable Calibration (Proprietary Models Only)", value=False, visible=False)


        def show_batch_calibration_mode(model_type):
            return gr.update(visible=model_type == "Proprietary Model")

        
        # 批量评估中的校准模式显示逻辑
        def update_batch_calibration_mode(model_type):
            if model_type == "Proprietary Model":
                return gr.update(visible=True, value=False)  # 显示校准模式并重置为 False
            return gr.update(visible=False, value=False)  # 隐藏校准模式并重置为 False

        
        model_type_selector.change(update_batch_calibration_mode, inputs=model_type_selector, outputs=batch_calibration_mode)

        batch_result_output = gr.Textbox(label="Batch Result", interactive=False)
        batch_evaluate_btn = gr.Button("Start Batch Evaluation", interactive=False)

        def batch_evaluation(file, path, mode, state, calibration_mode):
            model_type = state.get("model_type")
            if model_type == "Proprietary Model":
                model_name = state.get("model_name")
                if not model_name:
                    return "请先加载 API 模型"
                if calibration_mode:
                    return calibrated_evaluation_batch(file, path, mode, model_name=model_name)
                return evaluate_batch_with_api(file, path, mode, model_name)

            llm = state.get("llm")
            tokenizer = state.get("tokenizer")
            sampling_params = state.get("sampling_params")
            if llm is None or tokenizer is None or sampling_params is None:
                return "请先加载模型"
            if calibration_mode:
                raise "校准模式只能用于专有模型。"

            return evaluate_batch(file, path, mode, llm, tokenizer, sampling_params)

        batch_evaluate_btn.click(batch_evaluation, inputs=[file_input, save_path_input, batch_mode_selector, state, batch_calibration_mode], outputs=batch_result_output)
        model_load_output.change(enable_evaluate_button, inputs=model_load_output, outputs=batch_evaluate_btn)

    gr.Markdown("<div style='font-size: 18px; text-align: center;'>Designed by Hongli Zhou</div>")

if __name__ == "__main__":
    demo.launch(pwa=True)