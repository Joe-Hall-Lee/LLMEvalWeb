import pandas as pd
import json
import gradio as gr
import os
import sys

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
        if mode == "Direct Evaluation":
            return [
                {"role": "system", "content": "You are a helpful and precise assistant for checking the quality of the answer."},
                {"role": "user", "content": f"""[Question]\n{instruction}\n[The Start of Assistant 1's Answer]\n{answer1}\n[The End of Assistant 1's Answer]\n\n[The Start of Assistant 2's Answer]\n{answer2}\n[The End of Assistant 2's Answer]\n\nWe would like to request your feedback on the performance of two AI assistants in response to the user question displayed above.\nPlease rate the helpfulness, relevance, accuracy, level of details of their responses. Each assistant receives an overall score on a scale of 1 to 10, where a higher score indicates better overall performance.\nPlease first output a single line containing only two values indicating the scores for Assistant 1 and 2, respectively. The two scores are separated by a space. In the subsequent line, please provide a comprehensive explanation of your evaluation, avoiding any potential bias and ensuring that the order in which the responses were presented does not affect your judgment."""},
            ]
        elif mode == "Chain of Thought (CoT)":
            return [
                {"role": "system", "content": "You are a helpful and precise assistant for checking the quality of the answer using a chain of thought reasoning approach."},
                {"role": "user", "content": f"""[Question]\n{instruction}\n[The Start of Assistant 1's Answer]\n{answer1}\n[The End of Assistant 1's Answer]\n\n[The Start of Assistant 2's Answer]\n{answer2}\n[The End of Assistant 2's Answer]\n\nWe would like to request your feedback on the performance of two AI assistants in response to the user question displayed above.\nPlease rate the helpfulness, relevance, accuracy, level of details of their responses. Each assistant receives an overall score on a scale of 1 to 10, where a higher score indicates better overall performance.\nIn the first line, please provide a comprehensive explanation of your evaluation, avoiding any potential bias and ensuring that the order in which the responses were presented does not affect your judgment.\nIn the subsequent line, please output a single line containing only two values indicating the scores for Assistant 1 and 2, respectively. The two scores are separated by a space. There should be nothing on this line except two scores and a space."""},
            ]
        else:
            raise ValueError(f"Unsupported mode: {mode}")
            

def extract_scores(result, mode):
    try:
        if mode == "Direct Evaluation":
            return list(map(float, result.splitlines()[0].split()))
        elif mode == "Chain of Thought (CoT)":
            return list(map(float, result.splitlines()[-1].split()))
        else:
            raise ValueError("Unsupported mode for score extraction.")
    except (IndexError, ValueError):
        raise ValueError("Failed to parse scores from the evaluation result.")


def evaluate(instruction, answer1, answer2, mode, llm=None, tokenizer=None, sampling_params=None, model_name=None):
    if model_name in PROPRIETARY_MODELS.values():
        conversation = create_prompt(instruction, answer1, answer2, mode, model_name)
        response = call_model(conversation, model_name)
        if response:
            result = response.strip()
        else:
            return "API 请求失败", ""
    else:
        if llm is None or tokenizer is None or sampling_params is None:
            return "请先加载模型", ""
        conversation = create_prompt(instruction, answer1, answer2, mode, model_name)

        if tokenizer.chat_template is None:
            prompt_token_ids = tokenizer.encode(conversation)
        else:
            prompt_token_ids = tokenizer.apply_chat_template(conversation, add_generation_prompt=True)
        outputs = llm.generate(prompt_token_ids=prompt_token_ids, sampling_params=sampling_params)
        result = outputs[0].outputs[0].text.strip()

    try:
        scores = extract_scores(result, mode)
    except ValueError as e:
        scores = ""
    if len(scores) == 2:
        score1, score2 = scores
        verdict = "Assistant 1 is better " if score1 > score2 else ("Assistant 2 is better " if score2 > score1 else "Both assistants are equally good! ")
    else:
        verdict = "Error parsing scores. Please check the judge model output."

    full_prompt = "\n".join([msg["content"] for msg in create_prompt(instruction, answer1, answer2, mode)])

    details = f"""
<div class=\"details-section\">
<h3>‍👨🏽‍💻 User</h3>
<pre>{full_prompt.replace('>', '&gt;').replace('<', '&lt;').replace('\\n', '<br>')}</pre>
<h3>‍🧑‍⚖️ Judge Model</h3>
<pre>{result.replace('\\n', '<br>')}</pre>
</div>
"""
    return verdict, details


def evaluate_batch(file, output_path, mode, llm, tokenizer, sampling_params, model_name):
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
    for _, row in df.iterrows():
        instruction = row.get('instruction', '')
        answer1 = row.get('answer1', '')
        answer2 = row.get('answer2', '')

        if not instruction or not answer1 or not answer2:
            results.append("Invalid row: Missing data")
            continue

        try:
            verdict, _ = evaluate(
                instruction, answer1, answer2, mode,
                llm=llm, tokenizer=tokenizer, sampling_params=sampling_params, model_name=model_name
            )
            results.append(verdict)
        except Exception as e:
            results.append(f"Error: {str(e)}")

    output_df = pd.DataFrame({
        'Instruction': df.get('instruction', []),
        'Answer 1': df.get('answer1', []),
        'Answer 2': df.get('answer2', []),
        'Evaluation Result': results
    })

    try:
        output_df.to_csv(output_path, index=False, encoding='utf-8')
        return f"评估结果已保存到 {output_path}"
    except Exception as e:
        return f"保存文件时出错：{str(e)}"

def calibrated_evaluation(instruction, answer1, answer2, mode, model_name=None, llm=None, tokenizer=None, sampling_params=None):
    if not instruction or not answer1 or not answer2:
        raise ValueError("Instruction, Answer 1, and Answer 2 cannot be empty.")

    if model_name:
        def surface_quality_prompt(answer):
            return [
                {"role": "system", "content": "You are a meticulous evaluator whose task is to assess the superficial quality of an AI assistant's response, and you should focus specifically on language expression without considering the factual accuracy of the information provided."},
                {"role": "user", "content": f"""[The Start of Answer]\n{answer}\n[The End of Answer]\n\n[System]\nEvaluate the superficial quality of the provided answer in terms of linguistic expression and stylistic presentation. Provide a score between 1 and 10, where 10 signifies exceptional superficial articulation encompassing aspects such as lexical diversity, structural coherence, stylistic elegance, and overall fluidity. \nOn the first line, offer a detailed rationale for your score, explaining how well the answer demonstrates each assessed quality aspect. Your analysis should be thorough and impartial, focusing solely on superficial elements.\nOn the subsequent line, your rating should be presented as a numerical value without any other comments or explanations. There should be nothing on this line except a score."""}
            ]

        # 获取表面质量分数
        surface_score_1 = call_model(surface_quality_prompt(answer1), model_name).strip().splitlines()[-1].strip()
        surface_score_2 = call_model(surface_quality_prompt(answer2), model_name).strip().splitlines()[-1].strip()
    else:
        raise NotImplementedError("Calibrated evaluation can only be implemented for proprietary models.")

    surface_score_1 = float(surface_score_1)
    surface_score_2 = float(surface_score_2)

    # 获取未校准分数
    conversation = create_prompt(instruction, answer1, answer2, mode)
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
        verdict = "Assistant 1 is better " if adjusted_score1 > adjusted_score2 else ("Assistant 2 is better " if adjusted_score2 > adjusted_score1 else "Both assistants are equally good!")
    else:
        verdict = "Error parsing scores."

    # 构建完整提示和详情
    full_prompt = "\n".join([msg["content"] for msg in conversation])

    details = f"""
<div class="details-section">
    <h3>‍👨🏽‍💻 User</h3>
    <pre>{full_prompt.replace('>', '&gt;').replace('<', '&lt;').replace('\n', '<br>')}</pre>
    <h3>‍🧑‍⚖️ Judge Model (Original Evaluation)</h3>
    <pre>{result.replace('\n', '<br>')}</pre>
    <h3>🔧 Calibration Details</h3>
    <ul class="calibration-details">
        <li><b>Original Scores:</b> Assistant 1: {original_scores[0]:.2f}, Assistant 2: {original_scores[1]:.2f}</li>
        <li><b>Superficial Quality Scores:</b> Assistant 1: {surface_score_1:.2f}, Assistant 2: {surface_score_2:.2f}</li>
        <li><b>Adjusted Scores:</b> Assistant 1: {adjusted_score1:.2f}, Assistant 2: {adjusted_score2:.2f}</li>
    </ul>
    <p>Verdict: {verdict}</p>
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
            verdict, details = evaluate(
                instruction, answer1, answer2, mode,
                model_name=model_name
            )
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
