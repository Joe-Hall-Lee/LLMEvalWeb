import gradio as gr


def show_batch_calibration_mode(model_type):
    return gr.update(visible=model_type == "专有模型")


def show_calibration_mode(model_type):
    return gr.update(visible=model_type == "专有模型")