import pandas as pd
import json
import gradio as gr
import os
import sys

import torch
import gc
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np

# 添加到 sys.path 以导入其他模块
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from call_model import call_model
from config import FINETUNED_JUDGE_MODELS, PROPRIETARY_MODELS


def create_prompt(instruction, answer1, answer2, mode, model_name=None):
    """
    根据模型名称生成不同的 prompt 模板。
    """
    if not instruction or not answer1 or not answer2:
        raise ValueError("Instruction, Answer 1, and Answer 2 cannot be empty.")

    if model_name and "judgelm" in model_name.lower():
        return f"""You are a helpful and precise assistant for checking the quality of the answer.
[Question]
{instruction}

[The Start of Assistant 1's Answer]
{answer1}

[The End of Assistant 1's Answer]

[The Start of Assistant 2's Answer]
{answer2}

[The End of Assistant 2's Answer]

[System]
We would like to request your feedback on the performance of two AI assistants in response to the user question displayed above.
Please rate the helpfulness, relevance, accuracy, level of details of their responses. Each assistant receives an overall score on a scale of 1 to 10, where a higher score indicates better overall performance.
Please first output a single line containing only two values indicating the scores for Assistant 1 and 2, respectively. The two scores are separated by a space. In the subsequent line, please provide a comprehensive explanation of your evaluation, avoiding any potential bias and ensuring that the order in which the responses were presented does not affect your judgment.

### Response:"""
    else:
        # 对话式模板
        if mode == "直接评估":
            return [
                {"role": "system", "content": "You are a helpful and precise assistant for checking the quality of the answer."},
                {"role": "user", "content": f"""[Question]\n{instruction}\n[The Start of Assistant 1's Answer]\n{answer1}\n[The End of Assistant 1's Answer]\n\n[The Start of Assistant 2's Answer]\n{answer2}\n[The End of Assistant 2's Answer]\n\nWe would like to request your feedback on the performance of two AI assistants in response to the user question displayed above.\nPlease rate the helpfulness, relevance, accuracy, level of details of their responses. Each assistant receives an overall score on a scale of 1 to 10, where a higher score indicates better overall performance.\nPlease first output a single line containing only two values indicating the scores for Assistant 1 and 2, respectively. The two scores are separated by a space. In the subsequent line, please provide a comprehensive explanation of your evaluation, avoiding any potential bias and ensuring that the order in which the responses were presented does not affect your judgment."""},
            ]
        elif mode == "思维链":
            return [
                {"role": "system", "content": "You are a helpful and precise assistant for checking the quality of the answer using a chain of thought reasoning approach."},
                {"role": "user", "content": f"""[Question]\n{instruction}\n[The Start of Assistant 1's Answer]\n{answer1}\n[The End of Assistant 1's Answer]\n\n[The Start of Assistant 2's Answer]\n{answer2}\n[The End of Assistant 2's Answer]\n\nWe would like to request your feedback on the performance of two AI assistants in response to the user question displayed above.\nPlease rate the helpfulness, relevance, accuracy, level of details of their responses. Each assistant receives an overall score on a scale of 1 to 10, where a higher score indicates better overall performance.\nIn the first line, please provide a comprehensive explanation of your evaluation, avoiding any potential bias and ensuring that the order in which the responses were presented does not affect your judgment.\nIn the subsequent line, please output a single line containing only two values indicating the scores for Assistant 1 and 2, respectively. The two scores are separated by a space. There should be nothing on this line except two scores and a space."""},
            ]
        else:
            raise ValueError(f"Unsupported mode: {mode}")


def extract_scores(result, mode):
    try:
        if mode == "直接评估":
            return list(map(float, result.splitlines()[0].split()))
        elif mode == "思维链":
            return list(map(float, result.splitlines()[-1].split()))
        else:
            raise ValueError("Unsupported mode for score extraction.")
    except (IndexError, ValueError):
        raise ValueError("Failed to parse scores from the evaluation result.")


def evaluate(instruction, answer1, answer2, mode, state=None, model_name=None, proprietary_model=None):
    # 创建 prompt
    conversation = create_prompt(instruction, answer1, answer2, mode, model_name)
    print(conversation)
    if not proprietary_model:

        model = state.get("model")
        tokenizer = state.get("tokenizer")

        if model_name is None:
            return "请先加载模型", "", []

        # 检查 tokenizer 是否支持 chat_template
        if tokenizer.chat_template is None:
            # 如果不支持 chat_template，直接使用 tokenizer 处理 conversation
            inputs = tokenizer(conversation, return_tensors="pt").to(model.device)
            input_ids = inputs["input_ids"]
            full_prompt = conversation
        else:
            # 如果支持 chat_template，使用 apply_chat_template 处理 conversation
            # 确保 conversation 是一个列表，且每个元素是字典格式
            if not isinstance(conversation, list) or not all(isinstance(msg, dict) for msg in conversation):
                raise ValueError("Conversation must be a list of dictionaries with 'role' and 'content' keys.")
            
            full_prompt = "\n".join([msg["content"] for msg in create_prompt(instruction, answer1, answer2, mode)])
            
            # 使用 apply_chat_template 生成 token IDs
            input_ids = tokenizer.apply_chat_template(conversation, add_generation_prompt=True, return_tensors="pt").to(model.device)
            inputs = {"input_ids": input_ids}

        # 生成模型输出
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=2048,
                return_dict_in_generate=True,
                output_scores=True  # 返回每个 token 的 logits
            )

        # 提取生成的 token IDs
        generated_token_ids = outputs.sequences[0]  # 完整的生成序列（包括输入和输出）

        # 提取裁判模型的输出部分
        input_length = input_ids.shape[1]  # 输入部分的长度
        output_token_ids = generated_token_ids[input_length:]  # 输出部分的 token IDs

        # 解码裁判模型的输出
        result = tokenizer.decode(output_token_ids, skip_special_tokens=True)

        # 提取所有 token 的 logprobs
        logprobs = []
        for scores in outputs.scores:
            logits = scores.log_softmax(dim=-1)  # 计算 logprobs
            logprobs.append(logits)

        # 计算置信度
        confidence = calculate_confidence(logprobs)
        print(f"置信度: {confidence}")
    else:
        full_prompt = "\n".join([msg["content"] for msg in create_prompt(instruction, answer1, answer2, mode)])
        result = call_model(conversation, proprietary_model)
        logprobs = None

    try:
        scores = extract_scores(result, mode)
    except ValueError as e:
        scores = ""
    if len(scores) == 2:
        score1, score2 = scores
        verdict = "大模型 1 更好 " if score1 > score2 else ("大模型 2 更好 " if score2 > score1 else "两个大模型表现相当！")
    else:
        verdict = "解析分数失败。请检查评估模型的输出。"


    details = f"""
<div class=\"details-section\">
<h3>‍👨🏽‍💻 用户</h3>
<pre>{full_prompt.replace('>', '&gt;').replace('<', '&lt;').replace('\\n', '<br>')}</pre>
<h3>‍🧑‍⚖️ 裁判模型</h3>
<pre>{result.replace('\\n', '<br>')}</pre>
</div>
"""
    return verdict, details, logprobs  # 返回 logprobs


def evaluate_batch(file, output_path, mode, state):
    if file is None:
        return "请上传文件", ""

    try:
        if file.name.endswith('.csv'):
            df = pd.read_csv(file.name)
        elif file.name.endswith('.json'):
            with open(file.name, 'r', encoding='utf-8') as f:
                data = json.load(f)
            df = pd.DataFrame(data)
        else:
            return "仅支持 CSV 或 JSON 格式的文件", ""
    except (pd.errors.ParserError, json.JSONDecodeError) as e:
        return f"文件解析错误：{e}", ""
    except Exception as e:
        return f"读取文件时出错：{e}", ""

    results = []
    for _, row in df.iterrows():
        instruction = row.get('instruction', '')
        answer1 = row.get('answer1', '')
        answer2 = row.get('answer2', '')

        if not instruction or not answer1 or not answer2:
            results.append("无效行：数据缺失")
            continue

        try:
            verdict, _, _= evaluate(instruction, answer1, answer2, mode, state, proprietary_model=state.get("proprietary_model_name"))
            results.append(verdict)
        except Exception as e:
            results.append(f"错误：{str(e)}")

    output_df = pd.DataFrame({
        '指令': df.get('instruction', []),
        '答案 1': df.get('answer1', []),
        '答案 2': df.get('answer2', []),
        '评估结果': results
    })

    try:
        output_df.to_csv(output_path, index=False, encoding='utf-8')
        return f"评估结果已保存到 {output_path}", ""
    except Exception as e:
        return f"保存文件时出错：{str(e)}", ""


details_visible = False

def toggle_details():
    global details_visible
    details_visible = not details_visible
    return (
        gr.update(visible=details_visible),
        "隐藏详情" if details_visible else "显示详情",
    )

def calibrated_evaluation(instruction, answer1, answer2, mode, model_name=None):
    if not instruction or not answer1 or not answer2:
        raise ValueError("Instruction, Answer 1, and Answer 2 cannot be empty.")

    def surface_quality_prompt(answer):
        return [
            {"role": "system", "content": "You are a meticulous evaluator whose task is to assess the superficial quality of an AI assistant's response, and you should focus specifically on language expression without considering the factual accuracy of the information provided."},
            {"role": "user", "content": f"""[The Start of Answer]\n{answer}\n[The End of Answer]\n\n[System]\nEvaluate the superficial quality of the provided answer in terms of linguistic expression and stylistic presentation. Provide a score between 1 and 10, where 10 signifies exceptional superficial articulation encompassing aspects such as lexical diversity, structural coherence, stylistic elegance, and overall fluidity. \nOn the first line, offer a detailed rationale for your score, explaining how well the answer demonstrates each assessed quality aspect. Your analysis should be thorough and impartial, focusing solely on superficial elements.\nOn the subsequent line, your rating should be presented as a numerical value without any other comments or explanations. There should be nothing on this line except a score."""}
        ]

    # 获取表面质量分数
    surface_score_1 = call_model(surface_quality_prompt(answer1), model_name).strip().splitlines()[-1].strip()
    surface_score_2 = call_model(surface_quality_prompt(answer2), model_name).strip().splitlines()[-1].strip()

    surface_score_1 = float(surface_score_1)
    surface_score_2 = float(surface_score_2)

    # 获取未校准分数
    conversation = create_prompt(instruction, answer1, answer2, mode, model_name)
    response = call_model(conversation, model_name)
    if response:
        result = response.strip()
    else:
        return "API 请求失败", ""

    # 提取未校准分数
    original_scores = extract_scores(result, mode)
    if len(original_scores) == 2:
        adjusted_score1 = original_scores[0] - 0.5 * surface_score_1
        adjusted_score2 = original_scores[1] - 0.5 * surface_score_2
        verdict = "大模型 1 更好 " if adjusted_score1 > adjusted_score2 else ("大模型 2 更好 " if adjusted_score2 > adjusted_score1 else "两个大模型表现相当！")
    else:
        verdict = "Error parsing scores."

    # 构建完整提示和详情
    full_prompt = "\n".join([msg["content"] for msg in conversation])

    details = f"""
<div class="details-section">
    <h3>‍👨🏽‍💻 用户</h3>
    <pre>{full_prompt.replace('>', '&gt;').replace('<', '&lt;').replace('\n', '<br>')}</pre>
    <h3>‍🧑‍⚖️ 裁判模型（原评估结果）</h3>
    <pre>{result.replace('\n', '<br>')}</pre>
    <h3>🔧 校准详情</h3>
    <ul class="calibration-details">
        <li><b>原分数：</b>{original_scores[0]:.2f} {original_scores[1]:.2f}</li>
        <li><b>表面质量得分：</b>{surface_score_1:.2f} {surface_score_2:.2f}</li>
        <li><b>最终得分：</b>{adjusted_score1:.2f} {adjusted_score2:.2f}</li>
    </ul>
    <p>评估结果：{verdict}</p>
</div>
"""

    return verdict, details


details_visible = False

def toggle_details():
    global details_visible
    details_visible = not details_visible
    return (
        gr.update(visible=details_visible),
        "隐藏详情" if details_visible else "显示详情",
    )

def calibrated_evaluation_batch(file, output_path, mode, model_name=None):
    """
    批量执行校准评估
    """
    if file is None:
        return "请上传文件"

    try:
        if file.name.endswith('.csv'):
            df = pd.read_csv(file.name)
        elif file.name.endswith('.json'):
            with open(file.name, 'r', encoding='utf-8') as f:
                data = json.load(f)
            df = pd.DataFrame(data)
        else:
            return "仅支持 CSV 或 JSON 格式的文件"
    except (pd.errors.ParserError, json.JSONDecodeError) as e:
        return f"文件解析错误：{e}"
    except Exception as e:
        return f"读取文件时出错：{e}"

    results = []
    details_list = [] # 用于存储详细信息的列表
    for _, row in df.iterrows():
        instruction = row.get('instruction', '')
        answer1 = row.get('answer1', '')
        answer2 = row.get('answer2', '')

        if not instruction or not answer1 or not answer2:
            results.append("Invalid row: Missing data")
            details_list.append("") # 对应地添加空字符串
            continue

        try:
            verdict, details = calibrated_evaluation(instruction, answer1, answer2, mode, model_name=model_name)
            results.append(verdict)
            details_list.append(details) # 添加详细信息
        except Exception as e:
            results.append(f"Error: {str(e)}")
            details_list.append(f"<pre>Details: {str(e)}</pre>") # 错误信息也添加到详情中

    output_df = pd.DataFrame({
        'Instruction': df.get('instruction', []),
        'Answer 1': df.get('answer1', []),
        'Answer 2': df.get('answer2', []),
        'Evaluation Result': results,
        'Evaluation Details': details_list # 将详细信息添加到 DataFrame
    })

    try:
        output_df.to_csv(output_path, index=False, encoding='utf-8')
        return f"评估结果已保存到 {output_path}"
    except Exception as e:
        return f"保存文件时出错：{str(e)}"


def evaluate_batch_with_api(file, output_path, mode, model_name):
    """
    批量执行评估，仅支持通过 API 调用模型。
    """
    if file is None:
        return "请上传文件"

    try:
        if file.name.endswith('.csv'):
            df = pd.read_csv(file.name)
        elif file.name.endswith('.json'):
            with open(file.name, 'r', encoding='utf-8') as f:
                data = json.load(f)
            df = pd.DataFrame(data)
        else:
            return "仅支持 CSV 或 JSON 格式的文件"
    except (pd.errors.ParserError, json.JSONDecodeError) as e:
        return f"文件解析错误：{e}"
    except Exception as e:
        return f"读取文件时出错：{e}"

    results = []
    details_list = []
    for _, row in df.iterrows():
        instruction = row.get('instruction', '')
        answer1 = row.get('answer1', '')
        answer2 = row.get('answer2', '')

        if not instruction or not answer1 or not answer2:
            results.append("Invalid row: Missing data")
            details_list.append("")
            continue

        try:
            # 使用 API 模型进行评估
            verdict, details, _ = evaluate(instruction, answer1, answer2, mode, proprietary_model=model_name)
            results.append(verdict)
            details_list.append(details)
        except Exception as e:
            results.append(f"Error: {str(e)}")
            details_list.append(f"<pre>Details: {str(e)}</pre>")

    output_df = pd.DataFrame({
        'Instruction': df.get('instruction', []),
        'Answer 1': df.get('answer1', []),
        'Answer 2': df.get('answer2', []),
        'Evaluation Result': results,
        'Evaluation Details': details_list
    })

    try:
        output_df.to_csv(output_path, index=False, encoding='utf-8')
        return f"评估结果已保存到 {output_path}"
    except Exception as e:
        return f"保存文件时出错：{str(e)}"


def calculate_confidence(logprobs):
    """
    计算熵（基于概率分布的不确定性）
    :param logprobs: 包含每个生成步骤的 logprobs 的列表，每个元素是 (batch_size, vocab_size) 的 Tensor
    :return: 熵值（标量）
    """
    if not logprobs:
        return 0.0  # 如果没有 logprobs，返回默认值

    # 计算每个时间步的熵
    entropy_list = []
    for logprob in logprobs:
        # 将 logprobs 转换为概率
        probs = torch.exp(logprob)  # (batch_size, vocab_size)
        # 计算熵: -sum(p * log(p))
        entropy = -torch.sum(probs * logprob, dim=-1)  # (batch_size,)
        entropy_list.append(entropy)

    # 计算平均熵
    entropy_mean = torch.mean(torch.stack(entropy_list))  # 对所有时间步取平均
    return entropy_mean.item()  # 返回标量值