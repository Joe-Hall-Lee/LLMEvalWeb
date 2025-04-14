import gc
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import gradio as gr


def load_model(model_path, state):
    try:
        # 加载 tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        
        # 检查是否有可用的 GPU
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # 加载模型并显式移动到 GPU（如果可用）
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        ).to(device)  # 显式将模型移动到 GPU 或 CPU
        
        # 将模型和 tokenizer 存储到 state 中
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
    # 收集要卸载的模型名称
    model_names = []
    if state.get("finetuned_model_name"):
        model_names.append(state.get("finetuned_model_name"))
    if state.get("proprietary_model_name"):
        model_names.append(state.get("proprietary_model_name"))
    
    # 清理微调模型相关资源
    if state.get("model"):
        del state["model"]
        state["model"] = None
    if state.get("tokenizer"):
        del state["tokenizer"]
        state["tokenizer"] = None
    
    # 清理 GPU 内存
    if model_names:  # 仅在有模型时清理内存
        torch.cuda.empty_cache()
        gc.collect()
    
    # 清理状态
    state["finetuned_model_name"] = None
    state["proprietary_model_name"] = None
    
    # 返回卸载信息
    if not model_names:
        return "没有模型需要卸载"
    if len(model_names) == 1:
        return "模型 '{}' 已卸载".format(model_names[0])
    return "模型 {} 已卸载".format(", ".join("'{}'".format(name) for name in model_names))