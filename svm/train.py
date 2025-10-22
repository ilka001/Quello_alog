#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HRV情绪四分类SVM模型训练 (快速版本)
使用前7列作为特征，emotion作为标签进行四分类
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import os

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

class EmotionClassifier:
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.svm_model = None
        self.feature_names = None
        self.class_names = None
        
    def load_data(self, file_path):
        """加载数据"""
        print("正在加载数据...")
        df = pd.read_csv(file_path)
        print(f"数据形状: {df.shape}")
        print(f"情绪标签分布:\n{df['emotion'].value_counts()}")
        return df
    
    def preprocess_data(self, df):
        """数据预处理"""
        print("\n开始数据预处理...")
        
        # 提取特征和标签
        feature_columns = ['RMSSD', 'pNN58', 'SDNN', 'SD1', 'SD2', 'SD1_SD2', 'SampEn']
        self.feature_names = feature_columns
        
        X = df[feature_columns].values
        y = df['emotion'].values
        
        # 编码标签
        y_encoded = self.label_encoder.fit_transform(y)
        self.class_names = self.label_encoder.classes_
        
        print(f"特征数量: {X.shape[1]}")
        print(f"样本数量: {X.shape[0]}")
        print(f"类别数量: {len(self.class_names)}")
        print(f"类别名称: {self.class_names}")
        
        return X, y_encoded
    
    def split_and_scale_data(self, X, y, test_size=0.2, random_state=42):
        """数据分割和标准化"""
        print(f"\n数据分割 (测试集比例: {test_size})...")
        
        # 分割数据
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # 标准化特征
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        print(f"训练集形状: {X_train_scaled.shape}")
        print(f"测试集形状: {X_test_scaled.shape}")
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def train_svm(self, X_train, y_train):
        """训练SVM模型 (快速版本)"""
        print("\n开始训练SVM模型...")
        
        # 使用预定义的最佳参数组合
        print("使用RBF核SVM...")
        self.svm_model = SVC(
            kernel='rbf',
            C=10.0,
            gamma='scale',
            random_state=42
        )
        
        self.svm_model.fit(X_train, y_train)
        print("SVM模型训练完成！")
        
        return {'kernel': 'rbf', 'C': 10.0, 'gamma': 'scale'}
    
    def evaluate_model(self, X_test, y_test):
        """评估模型"""
        print("\n评估模型性能...")
        
        # 预测
        y_pred = self.svm_model.predict(X_test)
        
        # 准确率
        accuracy = accuracy_score(y_test, y_pred)
        print(f"测试集准确率: {accuracy:.4f}")
        
        # 详细分类报告
        print("\n分类报告:")
        print(classification_report(y_test, y_pred, target_names=self.class_names))
        
        # 混淆矩阵
        cm = confusion_matrix(y_test, y_pred)
        
        # 绘制混淆矩阵
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=self.class_names, yticklabels=self.class_names)
        plt.title('混淆矩阵')
        plt.xlabel('预测标签')
        plt.ylabel('真实标签')
        plt.tight_layout()
        plt.savefig('svm/confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return accuracy, cm
    
    def cross_validation(self, X, y, cv=5):
        """交叉验证"""
        print(f"\n进行{cv}折交叉验证...")
        
        # 标准化数据
        X_scaled = self.scaler.fit_transform(X)
        
        # 交叉验证
        cv_scores = cross_val_score(self.svm_model, X_scaled, y, cv=cv, scoring='accuracy')
        
        print(f"交叉验证得分: {cv_scores}")
        print(f"平均得分: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        return cv_scores
    
    def feature_importance_analysis(self, X, y):
        """特征重要性分析"""
        print("\n分析特征重要性...")
        
        # 使用线性核SVM获取特征权重
        linear_svm = SVC(kernel='linear', C=1.0, random_state=42)
        X_scaled = self.scaler.fit_transform(X)
        linear_svm.fit(X_scaled, y)
        
        # 获取特征权重
        feature_weights = np.abs(linear_svm.coef_)
        feature_importance = np.mean(feature_weights, axis=0)
        
        # 创建特征重要性DataFrame
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': feature_importance
        }).sort_values('importance', ascending=False)
        
        print("特征重要性排序:")
        print(importance_df)
        
        # 绘制特征重要性图
        plt.figure(figsize=(10, 6))
        sns.barplot(data=importance_df, x='importance', y='feature')
        plt.title('特征重要性')
        plt.xlabel('重要性')
        plt.tight_layout()
        plt.savefig('svm/feature_importance.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return importance_df
    
    def save_model(self, model_path='svm/emotion_svm_model.pkl'):
        """保存模型"""
        print(f"\n保存模型到: {model_path}")
        
        model_data = {
            'svm_model': self.svm_model,
            'scaler': self.scaler,
            'label_encoder': self.label_encoder,
            'feature_names': self.feature_names,
            'class_names': self.class_names
        }
        
        joblib.dump(model_data, model_path)
        print("模型保存完成！")
    
    def predict_sample(self, sample_data):
        """预测单个样本"""
        if isinstance(sample_data, list):
            sample_data = np.array(sample_data).reshape(1, -1)
        
        X_scaled = self.scaler.transform(sample_data)
        y_pred = self.svm_model.predict(X_scaled)
        y_pred_labels = self.label_encoder.inverse_transform(y_pred)
        
        # 获取预测概率
        y_prob = self.svm_model.decision_function(X_scaled)
        
        return y_pred_labels[0], y_prob[0]

def main():
    """主函数"""
    print("=== HRV情绪四分类SVM训练 (快速版本) ===")
    
    # 创建分类器实例
    classifier = EmotionClassifier()
    
    # 加载数据
    df = classifier.load_data(r'C:\Users\QAQ\Desktop\emotion\svm\test_processed.csv')
    
    # 数据预处理
    X, y = classifier.preprocess_data(df)
    
    # 分割和标准化数据
    X_train, X_test, y_train, y_test = classifier.split_and_scale_data(X, y)
    
    # 训练模型
    params = classifier.train_svm(X_train, y_train)
    
    # 评估模型
    test_accuracy, confusion_mat = classifier.evaluate_model(X_test, y_test)
    
    # 交叉验证
    cv_scores = classifier.cross_validation(X, y)
    
    # 特征重要性分析
    feature_importance = classifier.feature_importance_analysis(X, y)
    
    # 保存模型
    classifier.save_model()
    
    # 打印最终结果
    print("\n" + "="*50)
    print("训练完成！")
    print(f"使用参数: {params}")
    print(f"测试集准确率: {test_accuracy:.4f}")
    print(f"交叉验证平均得分: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    print("="*50)
    
    # 测试预测功能
    print("\n测试预测功能...")
    sample = X_test[0]  # 使用第一个测试样本
    prediction, decision_scores = classifier.predict_sample(sample)
    true_label = classifier.label_encoder.inverse_transform([y_test[0]])[0]
    
    print(f"真实标签: {true_label}")
    print(f"预测标签: {prediction}")
    print(f"决策函数得分: {decision_scores}")

if __name__ == "__main__":
    main()
