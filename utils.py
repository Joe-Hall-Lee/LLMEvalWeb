import gradio as gr
import os
import sys

from config import FINETUNED_JUDGE_MODELS, PROPRIETARY_MODELS
from webui.evaluation import evaluate, evaluate_batch, toggle_details, calibrated_evaluation, calibrated_evaluation_batch, evaluate_batch_with_api
from models.model import load_model, clear_model

# 启用评估按钮
def enable_evaluate_button(load_status):
    if "successfully" in load_status.lower():  # 判断加载状态是否成功
        return gr.update(interactive=True)
    return gr.update(interactive=False)


# 根据选择的模型类型动态更新模型选项
def update_model_choices(model_type):
    if model_type == "微调裁判模型":
        return gr.Dropdown(choices=list(FINETUNED_JUDGE_MODELS.keys()), value=list(FINETUNED_JUDGE_MODELS.keys())[0])
    else:
        return gr.Dropdown(choices=list(PROPRIETARY_MODELS.keys()), value=list(PROPRIETARY_MODELS.keys())[0])


def load_model_based_on_type(model_name, model_type, state):
    try:
        if model_type == "微调裁判模型":
            model_path = FINETUNED_JUDGE_MODELS[model_name]  # 使用字典访问
            state["model_name"] = model_path
            print(f"Loading model {model_name} from {model_path}")
            load_status, button_state = load_model(model_path, state)
            return load_status, gr.update(interactive=True)
        elif model_type == "专有模型":
            model_path = PROPRIETARY_MODELS[model_name]  # 使用字典访问
            state["model_name"] = model_path
            return f"API model {model_name} loaded successfully", gr.update(interactive=True)
        else:
            return "请选择模型类型", gr.update(interactive=True)
    except Exception as e:
        return f"加载模型失败: {str(e)}", gr.update(interactive=True)


# 手动评估中的校准模式显示逻辑
def update_calibration_mode(model_type):
    if model_type == "专有模型":
        return gr.update(visible=True, value=False)  # 显示校准模式并重置为 False
    return gr.update(visible=False, value=False)  # 隐藏校准模式并重置为 False


def show_calibration_mode(model_type):
    return gr.update(visible=model_type == "专有模型")

def manual_evaluate(instruction, answer1, answer2, mode, state, calibration_mode):
    model_name = state.get("model_name")
    model_type = state.get("model_type")
    if model_type == "专有模型":
        if not model_name:
            return "请先加载模型", "", gr.update(visible=False)
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

    return evaluate(instruction, answer1, answer2, mode, llm=llm, tokenizer=tokenizer, sampling_params=sampling_params, model_name=model_name)

def show_batch_calibration_mode(model_type):
    return gr.update(visible=model_type == "专有模型")


# 批量评估中的校准模式显示逻辑
def update_batch_calibration_mode(model_type):
    if model_type == "专有模型":
        return gr.update(visible=True, value=False)  # 显示校准模式并重置为 False
    return gr.update(visible=False, value=False)  # 隐藏校准模式并重置为 False


def batch_evaluation(file, path, mode, state, calibration_mode):
    model_type = state.get("model_type")
    if model_type == "专有模型":
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