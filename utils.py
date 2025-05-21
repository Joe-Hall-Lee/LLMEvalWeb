import gradio as gr
import os
import sys
from config import FINETUNED_JUDGE_MODELS, PROPRIETARY_MODELS
from webui.evaluation import evaluate, evaluate_batch, calibrated_evaluation, calibrated_evaluation_batch, evaluate_batch_with_api, calculate_confidence
import pandas as pd
import json
from modelscope import AutoModelForCausalLM, AutoTokenizer
import gc
import torch
import uuid
import tempfile

def enable_evaluate_button(load_status):
    if "成功" in load_status.lower():
        return gr.update(interactive=True)
    return gr.update(interactive=False)

def update_model_choices(model_type):
    if model_type == "微调裁判模型":
        return [
            gr.update(
                choices=list(FINETUNED_JUDGE_MODELS.keys()),
                value=list(FINETUNED_JUDGE_MODELS.keys())[0],
                visible=True
            ),
            gr.update(visible=False)
        ]
    else:
        return [
            gr.update(visible=False),
            gr.update(
                choices=list(PROPRIETARY_MODELS.keys()),
                value=list(PROPRIETARY_MODELS.keys())[0],
                visible=True
            )
        ]

def load_model_based_on_type(model_type, finetuned_model_name, proprietary_model_name, eval_mode, state):
    if not isinstance(state, dict):
        print(f"错误：state 不是字典，收到 {type(state)}: {state}")
        return "错误：内部状态错误，请刷新页面", gr.update(interactive=True)
    try:
        if eval_mode == "单模型评估":
            if model_type == "微调裁判模型":
                if not finetuned_model_name:
                    return "错误：未选择微调模型", gr.update(interactive=True)
                if finetuned_model_name not in FINETUNED_JUDGE_MODELS:
                    print(f"错误：无效的微调模型 {finetuned_model_name}")
                    return f"错误：无效的微调模型 {finetuned_model_name}", gr.update(interactive=True)
                model_path = FINETUNED_JUDGE_MODELS[finetuned_model_name]
                if not isinstance(model_path, str):
                    return f"错误：微调模型路径必须是字符串，收到 {type(model_path)}", gr.update(interactive=True)
                state["finetuned_model_name"] = finetuned_model_name
                state["proprietary_model_name"] = None
                state["model_type"] = model_type
                print(f"Loading model {finetuned_model_name} from {model_path}")
                load_status, button_state = load_model(model_path, state)
                return load_status, button_state
            elif model_type == "专有模型":
                if not proprietary_model_name:
                    return "错误：未选择专有模型", gr.update(interactive=True)
                if proprietary_model_name not in PROPRIETARY_MODELS:
                    return f"错误：无效的专有模型 {proprietary_model_name}", gr.update(interactive=True)
                model_path = PROPRIETARY_MODELS[proprietary_model_name]
                if not isinstance(model_path, str):
                    return f"错误：专有模型路径必须是字符串，收到 {type(model_path)}", gr.update(interactive=True)
                state["proprietary_model_name"] = proprietary_model_name
                state["finetuned_model_name"] = None
                state["model"] = None
                state["tokenizer"] = None
                state["model_type"] = model_type
                print(f"Initialized proprietary model {proprietary_model_name} with path {model_path}")
                return f"专有模型 {proprietary_model_name} 初始化成功！", gr.update(interactive=True)
            else:
                return "请选择有效模型类型", gr.update(interactive=True)
        elif eval_mode == "级联评估":
            if not finetuned_model_name or not proprietary_model_name:
                return "请选择微调裁判模型和专有模型", gr.update(interactive=True)
            finetuned_model_path = FINETUNED_JUDGE_MODELS.get(finetuned_model_name)
            if not finetuned_model_path:
                return f"错误：无效的微调模型 {finetuned_model_name}", gr.update(interactive=True)
            if not isinstance(finetuned_model_path, str):
                return f"错误：微调模型路径必须是字符串，收到 {type(finetuned_model_path)}", gr.update(interactive=True)
            proprietary_model_path = PROPRIETARY_MODELS.get(proprietary_model_name)
            if not proprietary_model_path:
                return f"错误：无效的专有模型 {proprietary_model_name}", gr.update(interactive=True)
            if not isinstance(proprietary_model_path, str):
                return f"错误：专有模型路径必须是字符串，收到 {type(proprietary_model_path)}", gr.update(interactive=True)
            state["finetuned_model_name"] = finetuned_model_name
            state["proprietary_model_name"] = proprietary_model_name
            state["model_type"] = model_type
            print(f"Loading finetuned model {finetuned_model_name} from {finetuned_model_path}")
            load_status, button_state = load_model(finetuned_model_path, state)
            print(f"Initialized proprietary model {proprietary_model_name} with path {proprietary_model_path}")
            return f"模型加载成功：\n\t- 微调裁判模型: {finetuned_model_name}\n\t- 专有模型: {proprietary_model_name}", button_state
        else:
            return "请选择评估模式", gr.update(interactive=True)
    except Exception as e:
        return f"加载模型失败: {str(e)}", gr.update(interactive=True)

def update_model_type(model_type, state):
    if not isinstance(state, dict):
        print(f"错误：state 不是字典，收到 {type(state)}: {state}")
        return state
    state["model_type"] = model_type
    return state

def update_calibration_mode(model_type):
    if model_type == "专有模型":
        return gr.update(visible=True, value=False)
    return gr.update(visible=False, value=False)

def manual_evaluate(instruction, answer1, answer2, mode, state, calibration_mode):
    if not isinstance(state, dict):
        return f"错误：state 不是字典，收到 {type(state)}", "", gr.update(visible=False)
    finetuned_model_name = state.get("finetuned_model_name")
    model_type = state.get("model_type")
    proprietary_model_name = state.get("proprietary_model_name")
    eval_mode = state.get("eval_mode")
    if eval_mode == "级联评估":
        llm = state.get("model")
        tokenizer = state.get("tokenizer")
        if llm is None or tokenizer is None:
            return "请先加载微调模型", "", gr.update(visible=False)
        verdict, details, logprobs, _, _ = evaluate(instruction, answer1, answer2, mode, state, finetuned_model_name)
        confidence = calculate_confidence(logprobs)
        threshold = state.get("confidence_threshold", 0.5)
        if confidence < threshold:
            if proprietary_model_name:
                if calibration_mode:
                    proprietary_verdict, proprietary_details = calibrated_evaluation(instruction, answer1, answer2, mode, model_name=proprietary_model_name)
                else:
                    proprietary_verdict, proprietary_details, _, _, _ = evaluate(instruction, answer1, answer2, mode, state=state, proprietary_model=proprietary_model_name)
                details = (
                    "<div class='details-section'>"
                    "<h3>级联评估详情</h3>"
                    f"<p>置信度低于阈值 ({confidence:.4f} < {threshold:.4f})，已调用专有模型重新评估。</p>"
                    f"{details}"
                    "<h3>‍🧑‍⚖️ 专有模型评估结果</h3>"
                    f"{proprietary_details}"
                    "</div>"
                )
                verdict = proprietary_verdict
            else:
                details = (
                    "<div class='details-section'>"
                    f"{details}"
                    f"<p>置信度低于阈值 ({confidence:.4f} < {threshold:.4f})，但未加载专有模型。</p>"
                    "</div>"
                )
        else:
            details = (
                "<div class='details-section'>"
                f"{details}"
                f"<p>置信度: {confidence:.4f} (高于阈值 {threshold:.4f})</p>"
                "</div>"
            )
        return verdict, details, gr.update(visible=True)
    else:
        if model_type == "专有模型":
            if not proprietary_model_name:
                return "请先加载模型", "", gr.update(visible=False)
            if calibration_mode:
                verdict, details = calibrated_evaluation(instruction, answer1, answer2, mode, model_name=proprietary_model_name)
                return verdict, details, gr.update(visible=True)
            verdict, details, _, _, _ = evaluate(instruction, answer1, answer2, mode, state=state, proprietary_model=proprietary_model_name)
            return verdict, details, gr.update(visible=True)
        llm = state.get("model")
        tokenizer = state.get("tokenizer")
        if llm is None or tokenizer is None:
            return "请先加载模型", "", gr.update(visible=False)
        verdict, details, _, _, _ = evaluate(instruction, answer1, answer2, mode, state=state, model_name=finetuned_model_name)
        return verdict, details, gr.update(visible=True)

def update_batch_calibration_mode(model_type):
    if model_type == "专有模型":
        return gr.update(visible=True, value=False)
    return gr.update(visible=False, value=False)

def batch_evaluation(file, mode, state, calibration_mode):
    if not isinstance(state, dict):
        return f"错误：state 不是字典，收到 {type(state)}", None
    eval_mode = state.get("eval_mode")
    if eval_mode == "级联评估":
        llm = state.get("model")
        tokenizer = state.get("tokenizer")
        threshold = state.get("confidence_threshold", 0.5)
        if llm is None or tokenizer is None:
            return "请先加载微调裁判模型", None
        proprietary_model_name = state.get("proprietary_model_name")
        if not proprietary_model_name:
            return "请先加载专有模型", None
        try:
            df = pd.read_csv(file.name) if file.name.endswith('.csv') else pd.DataFrame(json.load(open(file.name)))
        except Exception as e:
            return f"文件读取失败：{str(e)}", None
        results = []
        temp_dir = tempfile.gettempdir()
        output_filename = f"eval_report_{uuid.uuid4().hex[:8]}.csv"
        output_path = os.path.join(temp_dir, output_filename)
        for _, row in df.iterrows():
            instruction = row.get('instruction', '')
            answer1 = row.get('answer1', '')
            answer2 = row.get('answer2', '')
            if not all([instruction, answer1, answer2]):
                results.append("数据不完整")
                continue
            verdict, details, logprobs = evaluate(
                instruction, answer1, answer2, 
                mode, state, state.get("finetuned_model_name")
            )
            confidence = calculate_confidence(logprobs)
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
                details = (
                    "<div class='details-section'>"
                    f"{details}"
                    f"<p>置信度低于阈值 ({confidence:.4f} < {threshold:.4f})，已使用专有模型重新评估</p>"
                    f"{proprietary_details}"
                    "</div>"
                )
            else:
                details = (
                    "<div class='details-section'>"
                    f"{details}"
                    f"<p>置信度: {confidence:.4f} (高于阈值 {threshold:.4f})</p>"
                    "</div>"
                )
            results.append(verdict)
        output_df = pd.DataFrame({
            '指令': df.get('instruction', []),
            '答案 1': df.get('answer1', []),
            '答案 2': df.get('answer2', []),
            '评估结果': results
        })
        try:
            output_df.to_csv(output_path, index=False, encoding='utf-8')
            return f"评估完成，点击下方下载报告", output_path
        except Exception as e:
            return f"保存结果失败：{str(e)}", None
    else:
        model_type = state.get("model_type")
        if model_type == "专有模型":
            model_name = state.get("proprietary_model_name")
            if not model_name:
                return "请先加载专有模型", None
            if calibration_mode:
                return calibrated_evaluation_batch(file, mode, model_name=model_name)
            return evaluate_batch(file, mode, state)
        llm = state.get("model")
        tokenizer = state.get("tokenizer")
        if llm is None or tokenizer is None:
            return "请先加载模型", None
        if calibration_mode:
            return "校准模式只能用于专有模型", None
        return evaluate_batch(file, mode, state)

def update_eval_mode(mode, state):
    if not isinstance(state, dict):
        print(f"错误：state 不是字典，收到 {type(state)}: {state}")
        return [
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(choices=["微调裁判模型"], value="微调裁判模型"),
            gr.update(visible=True),
            gr.update(visible=False),
            gr.update(visible=True),
            gr.update(visible=False),
        ]
    state["eval_mode"] = mode
    return [
        gr.update(visible=mode == "单模型评估"),
        gr.update(visible=mode == "级联评估"),
        gr.update(visible=mode == "级联评估"),
        gr.update(choices=["微调裁判模型", "专有模型"] if mode == "单模型评估" else ["微调裁判模型"], value="微调裁判模型"),
        gr.update(visible=True),
        gr.update(visible=mode == "级联评估" or mode == "单模型评估"),
        gr.update(visible=True),
        gr.update(visible=mode == "级联评估" or mode == "单模型评估"),
    ]

def load_model(model_path, state):
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        ).to(device)
        state["model"] = model
        state["tokenizer"] = tokenizer
        return "模型加载成功！", gr.update(interactive=True)
    except RuntimeError as re:
        print(f"RuntimeError during model loading: {re}")
        if "out of memory" in str(re).lower():
            return "模型加载失败：内存不足。请释放内存或使用更小的模型。", gr.update(interactive=True)
        return f"模型加载失败：{re}", gr.update(interactive=True)
    except ValueError as ve:
        print(f"ValueError during model loading: {ve}")
        if "is not a local folder and is not a valid model identifier listed on 'https://huggingface.co/models'" in str(ve):
            return f"模型加载失败：请检查模型路径 '{model_path}' 是否正确，或模型是否已下载。", gr.update(interactive=True)
        return f"模型加载失败：{ve}", gr.update(interactive=True)
    except OSError as oe:
        print(f"OSError during model loading: {oe}")
        if "No such file or directory" in str(oe):
            return f"模型加载失败：路径 '{model_path}' 不存在。请检查路径是否正确。", gr.update(interactive=True)
        return f"模型加载失败：{oe}", gr.update(interactive=True)
    except Exception as e:
        return f"模型加载失败：未知错误：{e}", gr.update(interactive=True)
    finally:
        gc.collect()

def clear_model(state):
    model_names = []
    if state.get("finetuned_model_name"):
        model_names.append(state.get("finetuned_model_name"))
    if state.get("proprietary_model_name"):
        model_names.append(state.get("proprietary_model_name"))
    
    if state.get("model"):
        del state["model"]
        state["model"] = None
    if state.get("tokenizer"):
        del state["tokenizer"]
        state["tokenizer"] = None
    
    if model_names:
        torch.cuda.empty_cache()
        gc.collect()
    
    state["finetuned_model_name"] = None
    state["proprietary_model_name"] = None
    
    if not model_names:
        return "没有模型需要卸载"
    if len(model_names) == 1:
        return "模型 '{}' 已卸载".format(model_names[0])
    return "模型 {} 已卸载".format(", ".join("'{}'".format(name) for name in model_names))