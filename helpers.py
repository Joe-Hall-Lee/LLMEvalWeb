import gradio as gr

def show_calibration_mode(model_type, eval_mode):
    # 如果是“专有模型”或“级联评估”，则显示“推理策略”和“启用校准”
    visible = (model_type == "专有模型" or eval_mode == "级联评估")
    return gr.update(visible=visible), gr.update(visible=visible)

def show_batch_calibration_mode(model_type, eval_mode):
    print(f"show_batch_calibration_mode: {model_type}, {eval_mode}")
    # 如果是“专有模型”或“级联评估”，则显示“推理策略”和“启用校准”
    visible = (model_type == "专有模型" or eval_mode == "级联评估")
    return gr.update(visible=visible), gr.update(visible=visible)