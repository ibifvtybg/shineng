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
import xgboost as xgb

# 设置中文字体
font_path = "SimHei.ttf"
font_prop = FontProperties(fname=font_path, size=20)

# 确保matplotlib使用指定的字体
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
st.markdown('<div class="title">浦东新区监测站交通污染预测</div>', unsafe_allow_html=True)

# 加载XGBoost模型
try:
    model = joblib.load('XGBoost2020.pkl')
except Exception as e:
    st.write(f"<div style='color: red;'>Error loading model: {e}</div>", unsafe_allow_html=True)
    model = None

# 获取模型输入特征数量及顺序
model_input_features = ["CO", "FSP", "NO2", "O3", "RSP", "SO2"]
expected_feature_count = len(model_input_features)

# 定义空气质量类别映射
category_mapping = {
    5: '严重污染',
    4: '重度污染',
    3: '中度污染',
    2: '轻度污染',
    1: '良',
    0: '优'
}

# Streamlit界面设置
st.markdown('<div class="subheader">请填写以下信息以进行交通污染预测：</div>', unsafe_allow_html=True)

# 一氧化碳浓度
CO = st.number_input("一氧化碳的24小时平均浓度（毫克每立方米）：", min_value=0.0, value=0.0,
                    help="请输入该监测站检测到的一氧化碳在24小时内的平均浓度值，单位为毫克每立方米。")
if CO is None:
    st.warning("一氧化碳浓度输入为空，已将其从本次预测数据中删除。")
    CO = 0.0

# PM2.5浓度
FSP = st.number_input("PM2.5的24小时平均浓度（毫克每立方米）：", min_value=0.0, value=0.0,
                    help="请输入该监测站检测到的PM2.5在24小时内的平均浓度值，单位为毫克每立方米。")
if FSP is None:
    st.warning("PM2.5浓度输入为空，已将其从本次预测数据中删除。")
    FSP = 0.0

# 二氧化氮浓度
NO2 = st.number_input("二氧化氮的24小时平均浓度（毫克每立方米）：", min_value=0.0, value=0.0,
                    help="请输入该监测站检测到的二氧化氮在24小时内的可平均浓度值，单位为毫克每立方米。")
if NO2 is None:
    st.warning("二氧化氮浓度输入为空，已将其从本次预测数据中删除。")
    NO2 = 0.0

# 臭氧浓度
O3 = st.number_input("臭氧的24小时平均浓度（毫克每立方米）：", min_value=0.0, value=0.0,
                    help="请输入该监测站检测到的臭氧在24小时内的平均浓度值，单位为毫克每立方米。")
if O3 is None:
    st.warning("臭氧浓度输入为空，已将其从本次预测数据中删除。", min_value=0.0, value=0.0,
                    help="请输入该监测站检测到的臭氧在24小时内的平均浓度值，单位为毫克每立方米。")
if O3 is None:
    st.warning("臭氧浓度输入为空，已将其从本次预测数据中删除。")
    O3 = 0.0

# PM10浓度
RSP = st.number_input("PM10的24小时平均浓度（毫克每立方米）：", min_value=0.0, value=0.0,
                    help="请输入该监测站检测到的PM10在24小时内的平均浓度值，单位为毫克每立方米。")
if RSP is None:
    st.warning("PM10浓度输入为空，已将其从本次预测数据中删除。")
    RSP = 0.0

# 二氧化硫浓度
SO2 = st.number_input("二氧化硫的24小时平均浓度（毫克每立方米）：", min_value=0.0, value=0.0,
                    help="请输入该监测站检测到的二氧化硫在24小时内的平均浓度值，单位为毫克每立方米。")
if SO2 is None:
    st.warning("二氧化硫浓度输入为空，已将其从本次预测数据中删除。")
    SO2 = 0.0

def predict():
    try:
        # 检查模型是否加载成功
        if model is None:
            st.write("<div style='color: red;'>模型加载失败，无法进行预测。</div>", unsafe_allow_html=True)
            return

        # 获取用户输入并构建特征数组
        user_inputs = {
            "CO": int(CO),
            "FSP": int(FSP),
            "NO2": int(NO2),
            "O3": int(O3),
            "RSP": int(RSP),
            "SO2": int(SO2)
        }

        feature_values = [user_inputs[feature] for feature in model_input_features]
        features_array = np.array([feature_values])

        # 使用XGBoost模型进行预测
        predicted_class = model.predict(features_array)[0]
        predicted_proba = model.predict_proba(features_array)[0]

        # 显示预测结果
        st.markdown(f"<div class='prediction-result'>预测类别：{category_mapping[predicted_class]}</div>", unsafe_allow_html=True)

        # 根据预测结果生成建议
        probability = predicted_proba[predicted_class] * 100
        probability_str = " ".join([f"{category_mapping[i]}: {predicted_proba[i]*100:.1f}%" for i in range(len(category_mapping))])
        advice = {
                    '严重污染': f"建议：根据我们的库，该日空气质量为严重污染。模型预测该日为严重污染的概率为 {probability:.1f}%。建议采取防护措施，减少户外活动。",
                    '重度污染': f"建议：根据我们的库，该日空气质量为重度污染。模型预测该日为重度污染的概率为 {probability:.1f}%。建议减少外出，佩戴防护口罩。",
                    '中度污染': f"建议：根据我们的库，该日空气质量为中度污染。模型预测该日为中度污染的概率为 {probability:.1f}%。敏感人群应减少户外活动。",
                    '轻度污染': f"建议：根据我们的库，该日空气质量为轻度污染。模型预测该日为轻度污染的概率为 {probability:.1f}%。可以适当进行户外活动，但仍需注意防护。",
                    '良': f"建议：根据我们的库，此日空气质量为良。模型预测此日空气质量为良的概率为 {probability:.1f}%。可以正常进行户外活动。",
                    '优': f"建议：根据我们的库，该日空气质量为优。模型预测该日空气质量为优的概率为 {probability:.1f}%。空气质量良好，尽情享受户外时光。",
        }[category_mapping[predicted_class]]
        
        result_text = f"预测概率：{probability_str}<br>{advice}"
        st.markdown(f"<div class='advice-text'>{result_text}</div>", unsafe_allow_html=True)

        # 计算 SHAP 值
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(features_array)

        # 计算每个类别的特征贡献度
        importance_df = pd.DataFrame()
        for i in range(shap_values.shape[2]):  # 对每个类别进行计算
            importance = np.abs(shap_values[:, :, i]).mean(axis=0)
            importance_df[f'Class_{i}'] = importance

        importance_df.index = model_input_features

        # 类别映射
        type_mapping = {
             5: '严重污染',
             4: '重度污染',
             3: '中度污染',
             2: '轻度污染',
             1: '良',
             0: '优'
        }
        importance_df.columns = [type_mapping[i] for i in range(importance_df.shape[1])]

        # 获取指定类别的 SHAP 值贡献度
        predicted_class_name = category_mapping[predicted_class]  # 根据预测类别获取类别名称
        importances = importance_df[predicted_class_name]  # 提取 importance_df 中对应的类别列

        # 准备绘制瀑布图的数据
        feature_name_mapping = {
            "CO": "一氧化碳浓度",
            "FSP": "PM2.5浓度",
            "NO2": "二氧化氮浓度",
            "O3": "臭氧浓度",
            "RSP": "PM10浓度",
            "SO2": "二氧化硫浓度"
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
        st.write(f"<div style='color: red;'>Error in prediction: {e}</div>", unsafe_allow_html=True)

if st.button("预测", key="predict_button"):
    predict()

# 页脚
st.markdown('<div class="footer">© 2024 All rights reserved.</div>', unsafe_allow_html=True)   