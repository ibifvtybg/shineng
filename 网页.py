# -*- coding: utf-8 -*-
"""
Created on Wed Apr 16 14:41:50 2025

@author: 18657
"""

import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.font_manager import FontProperties
from xgboost import XGBClassifier

# 设置中文字体
font_path = "SimHei.ttf"
font_prop = FontProperties(fname=font_path, size=20)
plt.rcParams['font.sans-serif'] = [font_prop.get_name()]
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 添加蓝色主题的CSS样式，修复背景颜色问题
st.markdown("""
    <style>
   .main {
        background-color: #007BFF;
        background-image: url('https://www.transparenttextures.com/patterns/light_blue_fabric.png');
        color: #ffffff;
        font-family: 'Arial', sans-serif;
    }
   .title {
        font-size: 48px;
        color: #808080;
        font-weight: bold;
        text-align: center;
        margin-bottom: 30px;
    }
   .subheader {
        font-size: 28px;
        color: #99CCFF;
        margin-bottom: 25px;
        text-align: center;
        border-bottom: 2px solid #80BFFF;
        padding-bottom: 10px;
        margin-top: 20px;
    }
   .input-label {
        font-size: 18px;
        font-weight: bold;
        color: #ADD8E6;
        margin-bottom: 10px;
    }
   .footer {
        text-align: center;
        margin-top: 50px;
        font-size: 16px;
        color: #D8BFD8;
        background-color: #0056b3;
        padding: 20px;
        border-top: 1px solid #6A5ACD;
    }
   .button {
        background-color: #0056b3;
        border: none;
        color: white;
        padding: 12px 24px;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        font-size: 18px;
        margin: 20px auto;
        cursor: pointer;
        border-radius: 10px;
        box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.5);
        transition: background-color 0.3s, box-shadow 0.3s;
    }
   .button:hover {
        background-color: #003366;
        box-shadow: 0px 6px 10px rgba(0, 0, 0, 0.7);
    }
   .stSelectbox,.stNumberInput,.stSlider {
        margin-bottom: 20px;
    }
   .stSlider > div {
        padding: 10px;
        background: #E6E6FA;
        border-radius: 10px;
    }
   .prediction-result {
        font-size: 24px;
        color: #ffffff;
        margin-top: 30px;
        padding: 20px;
        border-radius: 10px;
        background: #4682B4;
        box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.3);
    }
   .advice-text {
        font-size: 20px;
        line-height: 1.6;
        color: #ffffff;
        background: #5DADE2;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.3);
        margin-top: 15px;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("<div class='main'>", unsafe_allow_html=True)

# 页面标题
st.markdown('<div class="title">老年人失能程度预测</div>', unsafe_allow_html=True)

# 加载XGBoost模型和标签编码器
try:
    model = joblib.load('final_xgb_model.pkl')
    label_encoder = joblib.load('label_encoder.pkl')
except Exception as e:
    st.write(f"<div style='color: red;'>Error loading model: {e}</div>", unsafe_allow_html=True)
    model = None

# 模型输入特征
model_input_features = ['体温', '脉搏', '收缩压', '舒张压', 'BMI', '吸烟', '饮酒', '高血压']
expected_feature_count = len(model_input_features)

# Streamlit界面设置
st.markdown('<div class="subheader">请填写以下健康指标以进行失能程度预测：</div>', unsafe_allow_html=True)

# 数值型特征输入
体温 = st.number_input("体温（℃）：", min_value=25.0, max_value=45.0, value=36.5, 
                     help="正常范围：36.1-37.2℃")
脉搏 = st.number_input("脉搏（次/分钟）：", min_value=30, max_value=200, value=70, 
                     help="正常范围：60-100次/分钟")
收缩压 = st.number_input("收缩压（mmHg）：", min_value=50, max_value=300, value=120, 
                        help="正常范围：90-139mmHg")
舒张压 = st.number_input("舒张压（mmHg）：", min_value=30, max_value=200, value=80, 
                        help="正常范围：60-89mmHg")
BMI = st.number_input("BMI（kg/m²）：", min_value=10.0, max_value=50.0, value=22.0, 
                    help="正常范围：18.5-23.9")

# 二分类特征输入
吸烟 = st.selectbox("是否吸烟：", ("否", "是"), index=0, help="吸烟状态")
饮酒 = st.selectbox("饮酒频率：", ("不饮", "偶饮", "常饮"), index=0, help="饮酒习惯")
高血压 = st.selectbox("是否患有高血压：", ("否", "是"), index=0, help="高血压病史")

def predict():
    try:
        if model is None:
            st.write("<div style='color: red;'>模型加载失败，无法进行预测。</div>", unsafe_allow_html=True)
            return

        # 构建特征字典
        user_inputs = {
            "体温": float(体温),
            "脉搏": float(脉搏),
            "收缩压": float(收缩压),
            "舒张压": float(舒张压),
            "BMI": float(BMI),
            "吸烟": 1 if 吸烟 == "是" else 0,
            "饮酒": 1 if 饮酒 in ["偶饮", "常饮"] else 0,
            "高血压": 1 if 高血压 == "是" else 0
        }

        # 生成特征数组
        feature_values = [user_inputs[feature] for feature in model_input_features]
        features_array = np.array([feature_values])

        # 模型预测
        y_pred = model.predict(features_array)
        y_proba = model.predict_proba(features_array)
        
        # 转换为原始标签
        predicted_class = y_pred[0]
        predicted_class_name = label_encoder.inverse_transform([predicted_class])[0]
        probas = {label: round(prob * 100, 2) for label, prob in zip(label_encoder.classes_, y_proba[0])}

        # 显示预测结果
        st.markdown(f"<div class='prediction-result'>失能风险等级：{predicted_class_name}</div>", unsafe_allow_html=True)
        
        # 生成建议
        advice = {
            '无失能': "建议：当前健康指标均在正常范围内，继续保持健康生活方式。",
            '轻度失能': "建议：部分指标异常，建议定期监测并咨询医生，调整生活习惯。",
            '中度失能': "建议：多项指标异常，存在一定失能风险，需及时就医检查并制定干预方案。",
            '重度失能': "建议：严重健康风险！请立即就医，进行全面身体检查和专业护理评估。"
        }[predicted_class_name]
        
        prob_text = " | ".join([f"{k}：{v}%" for k, v in probas.items()])
        result_text = f"预测概率：{prob_text}<br><br>{advice}"
        st.markdown(f"<div class='advice-text'>{result_text}</div>", unsafe_allow_html=True)

        # 计算 SHAP 值
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(features_array)

        # 计算每个类别的特征贡献度
        importance_df = pd.DataFrame()
        for i in range(len(shap_values)):  # 对每个类别进行计算
            importance = np.abs(shap_values[i]).mean(axis=0)
            importance_df[f'Class_{i}'] = importance

        importance_df.index = model_input_features
        # 修正类别映射，确保与标签编码器中的类别一致
        type_mapping = {i: label for i, label in enumerate(label_encoder.classes_)}
        importance_df = importance_df.rename(columns=type_mapping)

        # 打印importance_df的形状和内容
        st.write("importance_df的形状:", importance_df.shape)
        st.write("importance_df的内容:\n", importance_df)

        # 获取指定类别的 SHAP 值贡献度
        try:
            importances = importance_df[predicted_class_name]  # 提取 importance_df 中对应的类别列
        except KeyError as e:
            st.write(f"<div style='color: red;'>在importance_df中找不到对应的类别: {e}</div>", unsafe_allow_html=True)
            return

        # 准备绘制瀑布图的数据
        feature_name_mapping = {
            "体温": "体温",
            "脉搏": "脉搏",
            "收缩压": "收缩压",
            "舒张压": "舒张压",
            "BMI": "BMI",
            "吸烟": "吸烟",
            "饮酒": "饮酒",
            "高血压": "高血压"
        }
        features = [feature_name_mapping[f] for f in importances.index.tolist()]  # 获取特征名称
        contributions = importances.values  # 获取特征贡献度

        # 确保瀑布图的数据是按贡献度绝对值降序排列的
        sorted_indices = np.argsort(np.abs(contributions))[::-1]
        features_sorted = [features[i] for i in sorted_indices]
        contributions_sorted = contributions[sorted_indices]

        # 初始化绘图
        fig, ax = plt.subplots(figsize=(14, 8))

        # 初始化累积值
        start = 0
        prev_contributions = [start]  # 起始值为0

        # 计算每一步的累积值
        for i in range(1, len(contributions_sorted)):
            prev_contributions.append(prev_contributions[-1] + contributions_sorted[i - 1])

        # 绘制瀑布图
        for i in range(len(contributions_sorted)):
            color = '#ff5050' if contributions_sorted[i] < 0 else '#66b3ff'  # 负贡献使用红色，正贡献使用蓝色
            if i == len(contributions_sorted) - 1:
                # 最后一个条形带箭头效果，表示最终累积值
                ax.barh(features_sorted[i], contributions_sorted[i], left=prev_contributions[i], color=color, edgecolor='black', height=0.5, hatch='/')
            else:
                ax.barh(features_sorted[i], contributions_sorted[i], left=prev_contributions[i], color=color, edgecolor='black', height=0.5)

            # 在每个条形上显示数值
            plt.text(prev_contributions[i] + contributions_sorted[i] / 2, i, f"{contributions_sorted[i]:.2f}", 
                    ha='center', va='center', fontsize=10, fontproperties=font_prop, color='black')
            
        # 设置图表属性
        plt.title(f'预测类型为{predicted_class_name}时的特征贡献度瀑布图', size = 20, fontproperties=font_prop)
        plt.xlabel('贡献度 (SHAP 值)', fontsize=20, fontproperties=font_prop)
        plt.ylabel('特征', fontsize=20, fontproperties=font_prop)
        plt.yticks(size = 20, fontproperties=font_prop)
        plt.xticks(size = 20, fontproperties=font_prop)
        plt.grid(axis='x', linestyle='--', alpha=0.7)

        # 增加边距避免裁剪
        plt.xlim(left=0, right=max(prev_contributions) + max(contributions_sorted) * 1.0)
        fig.subplots_adjust(left=0.15, right=0.95, top=0.9, bottom=0.15)

        plt.tight_layout()

        # 保存并在 Streamlit 中展示
        plt.savefig("shap_waterfall_plot.png", bbox_inches='tight', dpi=1200)
        st.image("shap_waterfall_plot.png")

    except Exception as e:
        st.write(f"<div style='color: red;'>预测过程中出现错误：{str(e)}</div>", unsafe_allow_html=True)

if st.button("预测", key="predict_button", help="点击进行失能风险预测"):
    predict()

# 页脚
st.markdown('<div class="footer">© 2025 老年人健康管理系统. 保留所有权利.</div>', unsafe_allow_html=True)
