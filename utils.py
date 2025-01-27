import gradio as gr
import os
import sys

from config import FINETUNED_JUDGE_MODELS, PROPRIETARY_MODELS
from webui.evaluation import evaluate, evaluate_batch, toggle_details, calibrated_evaluation, calibrated_evaluation_batch, evaluate_batch_with_api, calculate_confidence
from models.model import load_model, clear_model
import pandas as pd
import json


# 启用评估按钮
def enable_evaluate_button(load_status):
    if "成功" in load_status.lower():  # 判断加载状态是否成功
        return gr.update(interactive=True)
    return gr.update(interactive=False)


# 根据选择的模型类型动态更新模型选项
def update_model_choices(model_type):
    if model_type == "微调裁判模型":
        return gr.Dropdown(choices=list(FINETUNED_JUDGE_MODELS.keys()), value=list(FINETUNED_JUDGE_MODELS.keys())[0])
    else:
        return gr.Dropdown(choices=list(PROPRIETARY_MODELS.keys()), value=list(PROPRIETARY_MODELS.keys())[0])


def load_model_based_on_type(finetuned_model_name, proprietary_model_name, eval_mode, state):
    try:
        if eval_mode == "单模型评估":
            # 单模型评估模式下，加载微调裁判模型或专有模型
            model_type = state.get("model_type")
            if model_type == "微调裁判模型":
                model_path = FINETUNED_JUDGE_MODELS[finetuned_model_name]  # 使用字典访问
                state["finetuned_model_name"] = model_path
                print(f"Loading model {finetuned_model_name} from {model_path}")
                load_status, button_state = load_model(model_path, state)
                return f"微调裁判模型 {finetuned_model_name} 加载成功！", gr.update(interactive=True)
            elif model_type == "专有模型":
                model_path = PROPRIETARY_MODELS[proprietary_model_name]  # 使用字典访问
                state["proprietary_model_name"] = model_path # 存储专有模型名称
                return f"专有模型 {proprietary_model_name} 加载成功！", gr.update(interactive=True)
            else:
                return "请选择模型类型", gr.update(interactive=True)
        elif eval_mode == "级联评估":
            # 级联评估模式下，同时加载微调裁判模型和专有模型
            if not finetuned_model_name:
                return "请选择微调裁判模型", gr.update(interactive=True)
            if not proprietary_model_name:
                return "请选择专有模型", gr.update(interactive=True)
            
            # 加载微调裁判模型
            finetuned_model_path = FINETUNED_JUDGE_MODELS[finetuned_model_name]
            state["finetuned_model_name"] = finetuned_model_path
            print(f"Loading finetuned model {finetuned_model_name} from {finetuned_model_path}")
            load_status, button_state = load_model(finetuned_model_path, state)
            
            # 加载专有模型
            proprietary_model_path = PROPRIETARY_MODELS[proprietary_model_name]
            state["proprietary_model_name"] = proprietary_model_path  # 存储专有模型名称
            print(f"Loading proprietary model {proprietary_model_name} from {proprietary_model_path}")
            
            return f"模型加载成功：\n\t- 微调裁判模型: {finetuned_model_name}\n\t- 专有模型: {proprietary_model_name}", gr.update(interactive=True)
        else:
            return "请选择评估模式", gr.update(interactive=True)
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
    finetuned_model_name = state.get("finetuned_model_name")
    model_type = state.get("model_type")
    proprietary_model_name = state.get("proprietary_model_name")
    eval_mode = state.get("eval_mode")  # 从 state 中获取 eval_mode

    if eval_mode == "级联评估":
        # 级联评估模式下，先使用微调裁判模型评估并计算置信度
        llm = state.get("model")
        tokenizer = state.get("tokenizer")

        if llm is None:
            return "请先加载模型", "", gr.update(visible=False)

        # 使用微调裁判模型进行评估
        verdict, details, logprobs = evaluate(instruction, answer1, answer2, mode, state, finetuned_model_name)  # 接收 logprobs

        # 计算置信度
        confidence = calculate_confidence(logprobs)  # 直接调用 calculate_confidence

        # 如果置信度低于阈值，调用专有模型重新评估
        threshold = state.get("confidence_threshold", 0.5)  # 默认阈值为 0.5
        if confidence < threshold:
                                
            # 调用专有模型重新评估
            proprietary_model_name = state.get("proprietary_model_name")
            if proprietary_model_name:
                if calibration_mode:
                    proprietary_verdict, proprietary_details = calibrated_evaluation(instruction, answer1, answer2, mode, model_name=proprietary_model_name)
                else:
                    proprietary_verdict, proprietary_details, _ = evaluate(instruction, answer1, answer2, mode, state=state, proprietary_model=proprietary_model_name
)
                details += f"""
<p>置信度低于阈值 ({confidence:.4f} < {threshold:.4f})，已调用专有模型重新评估。</p>
<h3>‍🧑‍⚖️ 专有模型评估结果</h3>
{proprietary_details}
"""
                verdict = proprietary_verdict  # 使用专有模型的评估结果
            else:
                details += f"\n<p>置信度低于阈值 ({confidence:.4f} < {threshold:.4f})，但未加载专有模型。</p>"
        else:
            details += f"\n<p>置信度: {confidence:.4f} (高于阈值 {threshold:.4f})</p>"

        return verdict, details, gr.update(visible=True)
    else:
        # 单模型评估模式
        if model_type == "专有模型":
            print(f"Loading proprietary model {proprietary_model_name}")
            if not proprietary_model_name:
                return "请先加载模型", "", gr.update(visible=False)
            if calibration_mode:
                return calibrated_evaluation(instruction, answer1, answer2, mode, model_name=proprietary_model_name)

            return evaluate(instruction, answer1, answer2, mode, state=state, proprietary_model=proprietary_model_name)

        llm = state.get("model")
        tokenizer = state.get("tokenizer")

        if llm is None or tokenizer is None:
            return "请先加载模型", "", gr.update(visible=False)

        if calibration_mode:
            return "校准模式只能用于专有模型。", "", gr.update(visible=False)

        return evaluate(instruction, answer1, answer2, mode, state=state, model_name=state.get("finetuned_model_name"))


def show_batch_calibration_mode(model_type):
    return gr.update(visible=model_type == "专有模型")


# 批量评估中的校准模式显示逻辑
def update_batch_calibration_mode(model_type):
    if model_type == "专有模型":
        return gr.update(visible=True, value=False)  # 显示校准模式并重置为 False
    return gr.update(visible=False, value=False)  # 隐藏校准模式并重置为 False


def batch_evaluation(file, path, mode, state, calibration_mode):
    eval_mode = state.get("eval_mode")
    
    if eval_mode == "级联评估":
        # 获取模型和阈值
        llm = state.get("model")
        tokenizer = state.get("tokenizer")
        threshold = state.get("confidence_threshold", 0.5)
        
        if llm is None or tokenizer is None:
            return "请先加载微调裁判模型"
            
        proprietary_model_name = state.get("proprietary_model_name")
        if not proprietary_model_name:
            return "请先加载专有模型"
            
        try:
            df = pd.read_csv(file.name) if file.name.endswith('.csv') else pd.DataFrame(json.load(open(file.name)))
        except Exception as e:
            return f"文件读取失败：{str(e)}"
            
        results = []
        details_list = []
        
        for _, row in df.iterrows():
            instruction = row.get('instruction', '')
            answer1 = row.get('answer1', '')
            answer2 = row.get('answer2', '')
            
            if not all([instruction, answer1, answer2]):
                results.append("数据不完整")
                continue
                
            # 先用微调裁判模型评估
            verdict, details, logprobs = evaluate(
                instruction, answer1, answer2, 
                mode, state, state.get("finetuned_model_name")
            )
            
            # 计算置信度
            confidence = calculate_confidence(logprobs)
            
            # 如果置信度低于阈值，使用专有模型重新评估
            if confidence < threshold:
                if calibration_mode:
                    proprietary_verdict, proprietary_details = calibrated_evaluation(
                        instruction, answer1, answer2, 
                        mode, model_name=proprietary_model_name
                    )
                else:
                    proprietary_verdict, proprietary_details, _ = evaluate(
                        instruction, answer1, answer2,
                        mode, state=state, 
                        proprietary_model=proprietary_model_name
                    )
                verdict = proprietary_verdict
                details += f"\n置信度低于阈值 ({confidence:.4f} < {threshold:.4f})，已使用专有模型重新评估"
            else:
                details += f"\n置信度: {confidence:.4f} (高于阈值 {threshold:.4f})"
                
            results.append(verdict)
            details_list.append(details)
            
        # 保存结果
        output_df = pd.DataFrame({
            '指令': df.get('instruction', []),
            '答案 1': df.get('answer1', []),
            '答案 2': df.get('answer2', []),
            '评估结果': results
        })
        
        try:
            output_df.to_csv(path, index=False, encoding='utf-8')
            return f"评估结果已保存到 {path}"
        except Exception as e:
            return f"保存结果失败：{str(e)}"
            
    else:
        # 原有的单模型评估逻辑
        model_type = state.get("model_type")
        if model_type == "专有模型":
            model_name = state.get("proprietary_model_name")
            if not model_name:
                return "请先加载专有模型"
            if calibration_mode:
                return calibrated_evaluation_batch(file, path, mode, model_name=model_name)
            return evaluate_batch_with_api(file, path, mode, model_name)

        llm = state.get("model")
        tokenizer = state.get("tokenizer")
        if llm is None or tokenizer is None:
            return "请先加载模型"
        if calibration_mode:
            return "校准模式只能用于专有模型"

        return evaluate_batch(file, path, mode, state)


# 绑定模型类型选择器的变更事件
def update_model_type(model_type, state):
    state["model_type"] = model_type  # 更新 state 中的 model_type
    return update_model_choices(model_type)


# 绑定评估模式选择器的变更事件
def update_eval_mode(mode, state):
    state["eval_mode"] = mode  # 更新 state 中的 eval_mode
    
    return [
        gr.update(visible=mode == "单模型评估"),  # 单模型评估时显示模型类型选择器
        gr.update(visible=mode == "级联评估"),  # 级联评估时显示专有模型选择器
        gr.update(visible=mode == "级联评估"),  # 级联评估时显示置信度阈值输入框
        gr.update(choices=["微调裁判模型", "专有模型"] if mode == "单模型评估" else ["微调裁判模型"], value="微调裁判模型"),  # 动态调整模型类型选择器
        gr.update(visible=True),  # 手动评估中的推理策略始终显示
        gr.update(visible=mode == "级联评估" or mode == "单模型评估"),  # 手动评估中的校准选项动态显示
        gr.update(visible=True),  # 批量评估中的推理策略始终显示
        gr.update(visible=mode == "级联评估" or mode == "单模型评估"),  # 批量评估中的校准选项动态显示
    ]