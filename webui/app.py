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

# Gradio 界面部分
with gr.Blocks(theme=Seafoam(), css=css) as demo:
    gr.Markdown("# 🌟 LLM Evaluation Web Application")
    gr.Markdown("### A web application for evaluating the outputs of large language models.")

    state = gr.State({"llm": None, "tokenizer": None, "sampling_params": None, "model_name": None, "model_type": "Fine-tuned Judge Model"})

    model_type_selector = gr.Radio(label="Select Model Type", choices=["Fine-tuned Judge Model", "Proprietary Model"], value="Fine-tuned Judge Model")
    model_selector = gr.Dropdown(label="Select Model", choices=FINETUNED_JUDGE_MODELS, value="Qwen2.5-3B-Instruct")
    model_load_output = gr.Textbox(label="Model Load Status", interactive=False)
    load_model_btn = gr.Button("Load Model")
    unload_model_btn = gr.Button("Unload Model")

    model_type_selector.change(update_model_options, inputs=model_type_selector, outputs=model_selector)

    load_model_btn.click(load_model_based_on_type, inputs=[model_selector, model_type_selector, state], outputs=[model_load_output, load_model_btn])

    unload_model_btn.click(clear_model, inputs=[state], outputs=model_load_output)

    # Manual Input Tab
    with gr.Tab("Manual Input"):
        instruction_input = gr.Textbox(label="Instruction", placeholder="Enter the instruction here...", lines=3)
        answer1_input = gr.Textbox(label="Assistant 1", placeholder="Enter the first answer here...", lines=3)
        answer2_input = gr.Textbox(label="Assistant 2", placeholder="Enter the second answer here...", lines=3)
        evaluation_mode_selector = gr.Radio(choices=["Direct Evaluation", "Chain of Thought (CoT)"], label="Evaluation Mode", value="Direct Evaluation")
        calibration_mode = gr.Checkbox(label="Enable Calibration (Proprietary Models Only)", value=False, visible=False) # 初始隐藏

        model_type_selector.change(update_calibration_mode, inputs=model_type_selector, outputs=calibration_mode)


        result_output = gr.Textbox(label="Evaluation Summary", interactive=False)
        details_output = gr.HTML("<div class='details-section'><h3>Details will appear here</h3></div>", visible=False, elem_classes=["details-section"])
        details_button = gr.Button("Show Details")
        evaluate_btn = gr.Button("Evaluate", interactive=False)

        

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

        
        model_type_selector.change(update_batch_calibration_mode, inputs=model_type_selector, outputs=batch_calibration_mode)

        batch_result_output = gr.Textbox(label="Batch Result", interactive=False)
        batch_evaluate_btn = gr.Button("Start Batch Evaluation", interactive=False)



        batch_evaluate_btn.click(batch_evaluation, inputs=[file_input, save_path_input, batch_mode_selector, state, batch_calibration_mode], outputs=batch_result_output)
        model_load_output.change(enable_evaluate_button, inputs=model_load_output, outputs=batch_evaluate_btn)

    gr.Markdown("<div style='font-size: 18px; text-align: center;'>Designed by Hongli Zhou</div>")

if __name__ == "__main__":
    demo.launch(pwa=True)