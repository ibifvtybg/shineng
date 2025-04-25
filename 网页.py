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
import xgbplot

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
吸烟 = st.radio("是否吸烟：", ("否", "是"), index=0, help="吸烟状态")
饮酒 = st.radio("饮酒频率：", ("不饮", "偶饮", "常饮"), index=0, help="饮酒习惯")
高血压 = st.radio("是否患有高血压：", ("否", "是"), index=0, help="高血压病史")

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
            "吸烟": 1 if吸烟 == "是" else 0,
            "饮酒": 1 if饮酒 in ["偶饮", "常饮"] else 0,
            "高血压": 1 if高血压 == "是" else 0
        }

        # 生成特征数组
        feature_values = [user_inputs[feature] for feature in model_input_features]
        features_array = np.array([feature_values])

        # 模型预测
        y_pred = model.predict(features_array)
        y_proba = model.predict_proba(features_array)
        
        # 转换为原始标签
        predicted_label = label_encoder.inverse_transform(y_pred)[0]
        probas = {label: round(prob*100, 1) for label, prob in zip(label_encoder.classes_, y_proba[0])}

        # 显示预测结果
        st.markdown(f"<div class='prediction-result'>失能风险等级：{predicted_label}</div>", unsafe_allow_html=True)
        
        # 生成建议
        advice = {
            '无失能': "建议：当前健康指标均在正常范围内，继续保持健康生活方式。",
            '轻度失能': "建议：部分指标异常，建议定期监测并咨询医生，调整生活习惯。",
            '中度失能': "建议：多项指标异常，存在一定失能风险，需及时就医检查并制定干预方案。",
            '重度失能': "建议：严重健康风险！请立即就医，进行全面身体检查和专业护理评估。"
        }[predicted_label]
        
        prob_text = " | ".join([f"{k}：{v}%" for k, v in probas.items()])
        result_text = f"预测概率：{prob_text}<br><br>{advice}"
        st.markdown(f"<div class='advice-text'>{result_text}</div>", unsafe_allow_html=True)

        # 计算并展示SHAP值
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(features_array)
        
        # 准备特征重要性数据
        feature_names = ['体温', '脉搏', '收缩压', '舒张压', 'BMI', '吸烟', '饮酒', '高血压']
        shap_importance = pd.DataFrame({
            '特征': feature_names,
            '重要性': np.abs(shap_values[0]).mean(axis=0)
        }).sort_values('重要性', ascending=False)

        # 绘制特征重要性图
        plt.figure(figsize=(12, 6))
        xgb.plot_importance(model, feature_names=feature_names, importance_type='gain', 
                          title="特征重要性分析", height=0.8, grid=False)
        plt.tight_layout()
        st.pyplot()

        # 绘制SHAP依赖图
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values[0], features_array, feature_names=feature_names, 
                         plot_type="bar", show=False)
        plt.title("特征贡献度分析", fontsize=16)
        plt.tight_layout()
        st.pyplot()

    except Exception as e:
        st.write(f"<div style='color: red;'>预测过程中出现错误：{str(e)}</div>", unsafe_allow_html=True)

if st.button("预测", key="predict_button", help="点击进行失能风险预测"):
    predict()

# 页脚
st.markdown('<div class="footer">© 2025 老年人健康管理系统. 保留所有权利.</div>', unsafe_allow_html=True)
