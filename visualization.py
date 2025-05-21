import os
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import gradio as gr
import re
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.font_manager import FontProperties


# 使用指定字体
plt.rcParams['axes.unicode_minus'] = False

# 报告存储目录，与 evaluation.py 保持一致
REPORT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "webui", "reports"))
os.makedirs(REPORT_DIR, exist_ok=True)  # 自动创建存储目录

def update_report_list():
    """
    刷新报告列表，返回可用的评估报告文件名供 Dropdown 选择。
    """
    try:
        report_files = [
            f for f in os.listdir(REPORT_DIR) 
            if f.startswith("eval_report_") and f.endswith(".csv")
        ]
        report_files.sort(reverse=True)  # 按时间倒序排列
        
        # 格式化显示名称，添加时间信息
        display_names = []
        for f in report_files:
            try:
                # 从文件名提取时间戳
                timestamp_str = re.search(r'(\d{8}_\d{6})', f).group(1)
                timestamp = datetime.strptime(timestamp_str, '%Y%m%d_%H%M%S')
                formatted_time = timestamp.strftime('%Y-%m-%d %H:%M:%S')
                display_name = f"{formatted_time} - {f}"
            except:
                display_name = f
            display_names.append(display_name)
        
        return gr.update(choices=display_names, value=display_names[0] if display_names else None, visible=True)
    except Exception as e:
        return gr.update(choices=[], visible=False, value=f"无法加载报告列表：{str(e)}")

def analyze_results(report_path):
    if not report_path:
        return "请先选择评估报告", None
    
    try:
        # 从显示名称中提取文件名
        report_filename = report_path.split(" - ")[-1]
        full_path = os.path.join(REPORT_DIR, report_filename)
        print("Full path:", full_path)  # 打印完整路径以检查
        
        df = pd.read_csv(full_path)
        print("df:", df)  # 打印数据以检查
        
        # 检查是否包含必要的列
        if 'score1' not in df.columns or 'score2' not in df.columns or 'winner' not in df.columns:
            return "评估报告缺少必要的列（score1, score2, winner）", None
        
        # 生成统计摘要
        stats_html = generate_stats_summary(df)
        print("stats_html:", stats_html)  # 打印统计摘要以检查
        
        # 生成模型对比分析图
        comp_plot = generate_comparison_plot(df)
        print("comp_plot:", comp_plot)  # 打印图表以检查
        
        return stats_html, comp_plot
    except Exception as e:
        return f"解析报告时出错: {str(e)}", None

def generate_stats_summary(df):
    # 生成统计摘要的 HTML
    stats_html = """
    <div class="stats-summary">
        <h3 style="color: #2c3e50; margin-bottom: 20px;">统计摘要</h3>
        <div class="stats-grid">
            <div class="stat-item">
                <div class="stat-value">{:.2f}</div>
                <div class="stat-label">模型1胜率</div>
            </div>
            <div class="stat-item">
                <div class="stat-value">{:.2f}</div>
                <div class="stat-label">模型2胜率</div>
            </div>
            <div class="stat-item">
                <div class="stat-value">{:.2f}</div>
                <div class="stat-label">平局率</div>
            </div>
            <div class="stat-item">
                <div class="stat-value">{:.2f}</div>
                <div class="stat-label">平均得分1</div>
            </div>
            <div class="stat-item">
                <div class="stat-value">{:.2f}</div>
                <div class="stat-label">平均得分2</div>
            </div>
        </div>
    </div>
    """.format(
        (df['winner'] == 'model1').mean() if 'winner' in df.columns else 0,
        (df['winner'] == 'model2').mean() if 'winner' in df.columns else 0,
        (df['winner'] == 'draw').mean() if 'winner' in df.columns else 0,
        df['score1'].mean() if 'score1' in df.columns else 0,
        df['score2'].mean() if 'score2' in df.columns else 0
    )
    return stats_html

def generate_comparison_plot(df):
    # 设置 Seaborn 样式
    sns.set(style="whitegrid", palette="pastel")
    
    # 创建子图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    # 胜率图
    win_rates = [
        (df['winner'] == 'model1').mean() if 'winner' in df.columns else 0,
        (df['winner'] == 'model2').mean() if 'winner' in df.columns else 0
    ]
    sns.barplot(
        x=['Model 1', 'Model 2'], 
        y=win_rates, 
        palette=['#1f77b4', '#ff7f0e'],  # 设置颜色
        ax=ax1,
        width=0.4  # 调整柱子宽度
    )
    ax1.set_title('Win Rates', fontsize=14, pad=20)
    ax1.set_ylabel('Win Rate', fontsize=12)
    ax1.set_ylim(0, 1)
    
    # 在柱状图上显示具体数值
    for p in ax1.patches:
        ax1.annotate(
            f'{p.get_height():.2%}', 
            (p.get_x() + p.get_width() / 2., p.get_height()), 
            ha='center', va='center', fontsize=12, color='black', xytext=(0, 5), 
            textcoords='offset points'
        )
    
    # 平均得分图
    avg_scores = [df['score1'].mean() if 'score1' in df.columns else 0, 
                  df['score2'].mean() if 'score2' in df.columns else 0]
    sns.barplot(
        x=['Model 1', 'Model 2'], 
        y=avg_scores, 
        palette=['#1f77b4', '#ff7f0e'],  # 设置颜色
        ax=ax2,
        width=0.4  # 调整柱子宽度
    )
    ax2.set_title('Average Scores', fontsize=14, pad=20)
    ax2.set_ylabel('Score', fontsize=12)
    ax2.set_ylim(0, max(avg_scores) + 1)
    
    # 在柱状图上显示具体数值
    for p in ax2.patches:
        ax2.annotate(
            f'{p.get_height():.2f}', 
            (p.get_x() + p.get_width() / 2., p.get_height()), 
            ha='center', va='center', fontsize=12, color='black', xytext=(0, 5), 
            textcoords='offset points'
        )
    
    # 调整布局
    plt.tight_layout()
    
    return fig