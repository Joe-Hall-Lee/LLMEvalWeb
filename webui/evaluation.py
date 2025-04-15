import pandas as pd
import json
import gradio as gr
import os
import sys
import torch
import gc
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
import tempfile
import uuid

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from call_model import call_model
from config import FINETUNED_JUDGE_MODELS, PROPRIETARY_MODELS

def create_prompt(instruction, answer1, answer2, mode, model_name=None):
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
    result = result.strip()
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
    try:
        conversation = create_prompt(instruction, answer1, answer2, mode, model_name)
        if not proprietary_model:
            model = state.get("model")
            tokenizer = state.get("tokenizer")
            if model is None or tokenizer is None or model_name is None:
                return "请先加载模型", "", []
            
            if tokenizer.chat_template is None:
                inputs = tokenizer(conversation, return_tensors="pt").to(model.device)
                input_ids = inputs["input_ids"]
                full_prompt = conversation
            else:
                if not isinstance(conversation, list) or not all(isinstance(msg, dict) for msg in conversation):
                    raise ValueError("Conversation must be a list of dictionaries with 'role' and 'content' keys.")
                full_prompt = "\n".join([msg["content"] for msg in create_prompt(instruction, answer1, answer2, mode)])
                input_ids = tokenizer.apply_chat_template(conversation, add_generation_prompt=True, return_tensors="pt").to(model.device)
                inputs = {"input_ids": input_ids}
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=2048,
                    return_dict_in_generate=True,
                    output_scores=True
                )
            
            generated_token_ids = outputs.sequences[0]
            input_length = input_ids.shape[1]
            output_token_ids = generated_token_ids[input_length:]
            result = tokenizer.decode(output_token_ids, skip_special_tokens=True)
            
            logprobs = []
            for scores in outputs.scores:
                logits = scores.log_softmax(dim=-1)
                logprobs.append(logits)
            
            confidence = calculate_confidence(logprobs)
            print(f"置信度: {confidence}")
        else:
            if not isinstance(proprietary_model, str):
                return f"错误：专有模型名称必须是字符串，收到 {type(proprietary_model)}", "", []
            if proprietary_model not in PROPRIETARY_MODELS:
                return f"错误：无效的专有模型 {proprietary_model}", "", []
            print(f"Calling call_model with proprietary_model: {proprietary_model}")
            full_prompt = "\n".join([msg["content"] for msg in create_prompt(instruction, answer1, answer2, mode)])
            result = call_model(conversation, PROPRIETARY_MODELS[proprietary_model])
            if result is None:
                return "错误：call_model 返回空结果", "", []
            print(f"call_model returned: {result}")
            logprobs = None
        
        scores = extract_scores(result, mode)
        if len(scores) == 2:
            score1, score2 = scores
            verdict = "大模型 1 更好" if score1 > score2 else ("大模型 2 更好" if score2 > score1 else "两个大模型表现相当！")
        else:
            verdict = "解析分数失败。请检查评估模型的输出。"
        
        details = (
            "<div class='details-section'>"
            "<h3>👨 用户</h3>"
            "<pre>%s</pre>"
            "<h3>⚖️ 裁判模型</h3>"
            "<pre>%s</pre>"
            "</div>"
        ) % (
            full_prompt.replace('>', '&gt;').replace('<', '&lt;').replace('\n', '<br>'),
            result.replace('\n', '<br>')
        )
        return verdict, details, logprobs
    except Exception as e:
        return f"评估失败: {str(e)}", "", []

def evaluate_batch(file, mode, state):
    if file is None:
        return "请上传文件", None
    
    try:
        temp_dir = tempfile.gettempdir()
        output_filename = f"eval_report_{uuid.uuid4().hex[:8]}.csv"
        output_path = os.path.join(temp_dir, output_filename)
        if file.name.endswith('.csv'):
            df = pd.read_csv(file.name)
        elif file.name.endswith('.json'):
            with open(file.name, 'r', encoding='utf-8') as f:
                data = json.load(f)
            df = pd.DataFrame(data)
        else:
            return "仅支持 CSV 或 JSON 格式的文件", None
    except (pd.errors.ParserError, json.JSONDecodeError) as e:
        return f"文件解析错误：{e}", None
    except Exception as e:
        return f"读取文件时出错：{e}", None
    
    results = []
    for _, row in df.iterrows():
        instruction = row.get('instruction', '')
        answer1 = row.get('answer1', '')
        answer2 = row.get('answer2', '')
        
        if not instruction or not answer1 or not answer2:
            results.append("无效行：数据缺失")
            continue
        
        try:
            if state.get("proprietary_model_name"):
                verdict, _, _ = evaluate(instruction, answer1, answer2, mode, state, proprietary_model=state.get("proprietary_model_name"))
            else:
                verdict, _, _ = evaluate(instruction, answer1, answer2, mode, state, model_name=state.get("finetuned_model_name"))
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
        return f"评估完成，点击下方下载报告", output_path
    except Exception as e:
        return f"保存文件时出错：{str(e)}", None

def calibrated_evaluation(instruction, answer1, answer2, mode, model_name=None):
    if not instruction or not answer1 or not answer2:
        raise ValueError("Instruction, Answer 1, and Answer 2 cannot be empty.")
    
    def surface_quality_prompt(answer):
        return [
            {"role": "system", "content": "You are a meticulous evaluator whose task is to assess the superficial quality of an AI assistant's response, and you should focus specifically on language expression without considering the factual accuracy of the information provided."},
            {"role": "user", "content": f"""[The Start of Answer]\n{answer}\n[The End of Answer]\n\n[System]\nEvaluate the superficial quality of the provided answer in terms of linguistic expression and stylistic presentation. Provide a score between 1 and 10, where 10 signifies exceptional superficial articulation encompassing aspects such as lexical diversity, structural coherence, stylistic elegance, and overall fluidity. \nOn the first line, offer a detailed rationale for your score, explaining how well the answer demonstrates each assessed quality aspect. Your analysis should be thorough and impartial, focusing solely on superficial elements.\nOn the subsequent line, your rating should be presented as a numerical value without any other comments or explanations. There should be nothing on this line except a score."""}
        ]
    
    try:
        if not isinstance(model_name, str):
            return f"错误：模型名称必须是字符串，收到 {type(model_name)}", ""
        surface_score_1 = call_model(surface_quality_prompt(answer1), PROPRIETARY_MODELS[model_name]).strip().splitlines()[-1].strip()
        surface_score_2 = call_model(surface_quality_prompt(answer2), PROPRIETARY_MODELS[model_name]).strip().splitlines()[-1].strip()
        try:
            surface_score_1 = float(surface_score_1)
        except (ValueError, TypeError):
            surface_score_1 = 0.0
        try:
            surface_score_2 = float(surface_score_2)
        except (ValueError, TypeError):
            surface_score_2 = 0.0
        
        conversation = create_prompt(instruction, answer1, answer2, mode, model_name)
        response = call_model(conversation, PROPRIETARY_MODELS[model_name])
        if response:
            result = response.strip()
        else:
            return "API 请求失败", ""
        
        original_scores = extract_scores(result, mode)
        if len(original_scores) == 2:
            adjusted_score1 = original_scores[0] - 0.8 * surface_score_1
            adjusted_score2 = original_scores[1] - 0.8 * surface_score_2
            verdict = "大模型 1 更好" if adjusted_score1 > adjusted_score2 else ("大模型 2 更好" if adjusted_score2 > adjusted_score1 else "两个大模型表现相当！")
        else:
            verdict = "解析分数失败"
        
        full_prompt = "\n".join([msg["content"] for msg in conversation])
        formatted_prompt = full_prompt.replace('>', '&gt;').replace('<', '&lt;').replace('\n', '<br>')
        formatted_result = result.replace('\n', '<br>')
        
        details = """
        <div class="details-section">
            <h3>‍👨🏽‍💻 用户</h3>
            <pre>{prompt}</pre>
            <h3>‍🧑‍⚖️ 裁判模型（原评估结果）</h3>
            <pre>{result}</pre>
            <h3>🔧 校准详情</h3>
            <ul class="calibration-details">
                <li><b>原分数：</b>{score1:.2f} {score2:.2f}</li>
                <li><b>表面质量得分：</b>{surf1:.2f} {surf2:.2f}</li>
                <li><b>最终得分：</b>{final1:.2f} {final2:.2f}</li>
            </ul>
            <p>评估结果：{verdict}</p>
        </div>
        """.format(
            prompt=formatted_prompt,
            result=formatted_result,
            score1=original_scores[0],
            score2=original_scores[1],
            surf1=surface_score_1,
            surf2=surface_score_2,
            final1=adjusted_score1,
            final2=adjusted_score2,
            verdict=verdict
        )
        return verdict, details
    except Exception as e:
        return f"校准评估失败: {str(e)}", ""

def calibrated_evaluation_batch(file, mode, model_name=None):
    if file is None:
        return "请上传文件", None
    
    try:
        temp_dir = tempfile.gettempdir()
        output_filename = f"eval_report_{uuid.uuid4().hex[:8]}.csv"
        output_path = os.path.join(temp_dir, output_filename)
        if file.name.endswith('.csv'):
            df = pd.read_csv(file.name)
        elif file.name.endswith('.json'):
            with open(file.name, 'r', encoding='utf-8') as f:
                data = json.load(f)
            df = pd.DataFrame(data)
        else:
            return "仅支持 CSV 或 JSON 格式的文件", None
    except (pd.errors.ParserError, json.JSONDecodeError) as e:
        return f"文件解析错误：{e}", None
    except Exception as e:
        return f"读取文件时出错：{e}", None
    
    results = []
    for _, row in df.iterrows():
        instruction = row.get('instruction', '')
        answer1 = row.get('answer1', '')
        answer2 = row.get('answer2', '')
        
        if not instruction or not answer1 or not answer2:
            results.append("无效行：数据缺失")
            continue
        
        try:
            verdict, _ = calibrated_evaluation(instruction, answer1, answer2, mode, model_name=model_name)
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
        return f"评估完成，点击下方下载报告", output_path
    except Exception as e:
        return f"保存文件时出错：{str(e)}", None

def evaluate_batch_with_api(file, mode, model_name):
    if file is None:
        return "请上传文件", None
    
    try:
        temp_dir = tempfile.gettempdir()
        output_filename = f"eval_report_{uuid.uuid4().hex[:8]}.csv"
        output_path = os.path.join(temp_dir, output_filename)
        if file.name.endswith('.csv'):
            df = pd.read_csv(file.name)
        elif file.name.endswith('.json'):
            with open(file.name, 'r', encoding='utf-8') as f:
                data = json.load(f)
            df = pd.DataFrame(data)
        else:
            return "仅支持 CSV 或 JSON 格式的文件", None
    except (pd.errors.ParserError, json.JSONDecodeError) as e:
        return f"文件解析错误：{e}", None
    except Exception as e:
        return f"读取文件时出错：{e}", None
    
    results = []
    for _, row in df.iterrows():
        instruction = row.get('instruction', '')
        answer1 = row.get('answer1', '')
        answer2 = row.get('answer2', '')
        
        if not instruction or not answer1 or not answer2:
            results.append("无效行：数据缺失")
            continue
        
        try:
            verdict, _, _ = evaluate(instruction, answer1, answer2, mode, proprietary_model=model_name)
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
        return f"评估完成，点击下方下载报告", output_path
    except Exception as e:
        return f"保存文件时出错：{str(e)}", None

def calculate_confidence(logprobs):
    if not logprobs:
        return 0.0
    
    entropy_list = []
    for logprob in logprobs:
        probs = torch.exp(logprob)
        entropy = -torch.sum(probs * logprob, dim=-1)
        entropy_list.append(entropy)
    
    entropy_mean = torch.mean(torch.stack(entropy_list))
    return entropy_mean.item()