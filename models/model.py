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
        state["model_name"] = model_path
        return "Model loaded successfully!", gr.update(interactive=True)
    except RuntimeError as re:
        print(f"RuntimeError during model loading: {re}")
        if "out of memory" in str(re).lower():
            return "Model loading failed: Out of memory. Please free up memory or use a smaller model.", gr.update(interactive=True)
        return f"Model loading failed: {re}", gr.update(interactive=True)
    except ValueError as ve:
        print(f"ValueError during model loading: {ve}")
        if "is not a local folder and is not a valid model identifier listed on 'https://huggingface.co/models'" in str(ve):
            return f"Model loading failed: Please check if the model path '{model_path}' is correct, or if the model has been downloaded.", gr.update(interactive=True)
        return f"Model loading failed: {ve}", gr.update(interactive=True)
    except OSError as oe:
        print(f"OSError during model loading: {oe}")
        if "No such file or directory" in str(oe):
            return f"Model loading failed: Specified path '{model_path}' does not exist. Please check if the path is correct.", gr.update(interactive=True)
        return f"Model loading failed: {oe}", gr.update(interactive=True)
    except Exception as e:
        return f"Unknown error occurred during model loading: {e}", gr.update(interactive=True)
    finally:
        gc.collect()



def clear_model(state):
    model_name = state.get("model_name")
    if state.get("llm"):
        del state["llm"]
        state["llm"] = None
    if state.get("tokenizer"):
        del state["tokenizer"]
        state["tokenizer"] = None
    state["sampling_params"] = None

    # Explicitly clear any leftover GPU memory
    torch.cuda.empty_cache()
    gc.collect()

    state["model_name"] = None
    return f"Model '{model_name}' unloaded" if model_name else "No model to unload" # 返回英文卸载信息