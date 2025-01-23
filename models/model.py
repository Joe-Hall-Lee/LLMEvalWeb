import gc
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
import gradio as gr
import torch


def load_model(model_path, state):
    try:
        llm = LLM(model=model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        sampling_params = SamplingParams(temperature=0.7, max_tokens=1024)
        state["llm"] = llm
        state["tokenizer"] = tokenizer
        state["sampling_params"] = sampling_params
        return "模型加载成功", gr.update(interactive=True)
    except RuntimeError as re:
        print(f"RuntimeError during model loading: {re}")
        if "out of memory" in str(re).lower():
            return "模型加载失败：内存不足，请释放显存或使用更小的模型。", gr.update(interactive=True)
        return f"模型加载失败：{re}", gr.update(interactive=True)
    except ValueError as ve:
        print(f"ValueError during model loading: {ve}")
        if "is not a local folder and is not a valid model identifier listed on 'https://huggingface.co/models'" in str(ve):
            return f"模型加载失败: 请检查模型路径 '{model_path}' 是否正确，或者模型是否已下载。", gr.update(interactive=True)
        return f"模型加载失败: {ve}", gr.update(interactive=True)
    except OSError as oe:
        print(f"OSError during model loading: {oe}")
        if "No such file or directory" in str(oe):
            return f"模型加载失败: 指定路径 '{model_path}' 不存在。请检查路径是否正确。", gr.update(interactive=True)
        return f"模型加载失败: {oe}", gr.update(interactive=True)
    except Exception as e:
        return f"模型加载时发生未知错误：{e}", gr.update(interactive=True)
    finally:
        gc.collect()



def clear_model(state):
    if state.get("model_name"):
        state["model_name"] = None
    if state.get("llm"):  # 使用 .get() 方法，避免 KeyError
        del state["llm"]  # Delete the model object to free up memory
    if state.get("tokenizer"):
        del state["tokenizer"]
    state["sampling_params"] = None

    # Explicitly clear any leftover GPU memory
    torch.cuda.empty_cache()
    gc.collect()  # Perform garbage collection
    state["llm"] = None
    state["tokenizer"] = None
    return "模型已卸载"