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
    model_name = state.get("finetuned_model_name")
    if state.get("model"):
        del state["model"]
        state["model"] = None
    if state.get("tokenizer"):
        del state["tokenizer"]
        state["tokenizer"] = None

    # 显式清理 GPU 内存
    torch.cuda.empty_cache()
    gc.collect()

    state["finetuned_model_name"] = None
    return f"模型 '{model_name}' 已卸载" if model_name else "没有模型需要卸载"