#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Delta特征与情绪标签相关性分析
分析HRV delta特征与情绪分类的相关性，验证分类模型的可行性
"""

import csv
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

class DeltaFeatureAnalyzer:
    def __init__(self, csv_file: str = "hrv_emotion_delta_results.csv"):
        """
        初始化Delta特征分析器
        
        Args:
            csv_file: CSV文件路径
        """
        self.csv_file = r"C:\Users\QAQ\Desktop\emotion\namecsv\hrv_emotion_delta_results.csv"
        self.data = None
        self.features = ['delta_RMSSD', 'delta_pNN58', 'delta_SDNN', 'delta_SD1', 'delta_SD2', 'delta_SD1_SD2']
        self.emotions = ['悲伤', '愉悦', '焦虑']
        
    def load_data(self):
        """加载数据"""
        try:
            self.data = pd.read_csv(self.csv_file)
            print(f"成功加载数据: {len(self.data)} 条记录")
            print(f"特征列: {self.features}")
            print(f"情绪类别: {self.emotions}")
            return True
        except Exception as e:
            print(f"加载数据失败: {e}")
            return False
    
    def clean_data(self):
        """清理数据"""
        if self.data is None:
            return False
        
        # 移除空值行
        initial_count = len(self.data)
        self.data = self.data.dropna()
        final_count = len(self.data)
        
        print(f"数据清理: {initial_count} -> {final_count} 条记录")
        
        # 检查情绪分布
        emotion_counts = self.data['emotion'].value_counts()
        print("\n情绪分布:")
        for emotion, count in emotion_counts.items():
            print(f"  {emotion}: {count} 条")
        
        return True
    
    def analyze_feature_distribution(self):
        """分析特征分布"""
        print("\n" + "="*60)
        print("特征分布分析")
        print("="*60)
        
        for feature in self.features:
            print(f"\n{feature}:")
            values = self.data[feature].dropna()
            if len(values) > 0:
                print(f"  均值: {values.mean():.4f}")
                print(f"  标准差: {values.std():.4f}")
                print(f"  最小值: {values.min():.4f}")
                print(f"  最大值: {values.max():.4f}")
                print(f"  中位数: {values.median():.4f}")
            else:
                print("  无有效数据")
    
    def analyze_emotion_feature_correlation(self):
        """分析情绪与特征的相关性"""
        print("\n" + "="*60)
        print("情绪-特征相关性分析")
        print("="*60)
        
        # 为每个情绪计算特征均值
        emotion_stats = {}
        for emotion in self.emotions:
            emotion_data = self.data[self.data['emotion'] == emotion]
            if len(emotion_data) > 0:
                emotion_stats[emotion] = {}
                for feature in self.features:
                    values = emotion_data[feature].dropna()
                    if len(values) > 0:
                        emotion_stats[emotion][feature] = {
                            'mean': values.mean(),
                            'std': values.std(),
                            'count': len(values)
                        }
        
        # 打印每个情绪的特征统计
        for emotion in self.emotions:
            if emotion in emotion_stats:
                print(f"\n{emotion} 情绪特征统计:")
                for feature in self.features:
                    if feature in emotion_stats[emotion]:
                        stats = emotion_stats[emotion][feature]
                        print(f"  {feature}: 均值={stats['mean']:+.4f}, 标准差={stats['std']:.4f}, 样本数={stats['count']}")
        
        # 进行ANOVA检验
        print(f"\nANOVA检验 (F统计量和p值):")
        for feature in self.features:
            groups = []
            for emotion in self.emotions:
                emotion_data = self.data[self.data['emotion'] == emotion]
                values = emotion_data[feature].dropna()
                if len(values) > 0:
                    groups.append(values)
            
            if len(groups) >= 2:
                f_stat, p_value = stats.f_oneway(*groups)
                print(f"  {feature}: F={f_stat:.4f}, p={p_value:.4f}")
                if p_value < 0.05:
                    print(f"    -> 显著差异 (p < 0.05)")
                else:
                    print(f"    -> 无显著差异 (p >= 0.05)")
    
    def analyze_feature_importance(self):
        """分析特征重要性"""
        print("\n" + "="*60)
        print("特征重要性分析")
        print("="*60)
        
        # 准备数据
        X = self.data[self.features].fillna(0)  # 用0填充缺失值
        y = self.data['emotion']
        
        # 使用随机森林计算特征重要性
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X, y)
        
        # 获取特征重要性
        feature_importance = pd.DataFrame({
            'feature': self.features,
            'importance': rf.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("随机森林特征重要性:")
        for _, row in feature_importance.iterrows():
            print(f"  {row['feature']}: {row['importance']:.4f}")
        
        # 使用SelectKBest进行特征选择
        selector = SelectKBest(score_func=f_classif, k='all')
        selector.fit(X, y)
        
        feature_scores = pd.DataFrame({
            'feature': self.features,
            'f_score': selector.scores_,
            'p_value': selector.pvalues_
        }).sort_values('f_score', ascending=False)
        
        print(f"\nF检验特征得分:")
        for _, row in feature_scores.iterrows():
            print(f"  {row['feature']}: F={row['f_score']:.4f}, p={row['p_value']:.4f}")
        
        return feature_importance, feature_scores
    
    def evaluate_classification_performance(self):
        """评估分类性能"""
        print("\n" + "="*60)
        print("分类性能评估")
        print("="*60)
        
        # 准备数据
        X = self.data[self.features].fillna(0)
        y = self.data['emotion']
        
        # 标准化特征
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # 使用随机森林进行分类
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        
        # 交叉验证
        cv_scores = cross_val_score(rf, X_scaled, y, cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42))
        
        print(f"交叉验证准确率: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        print(f"各折准确率: {cv_scores}")
        
        # 训练完整模型并生成详细报告
        rf.fit(X_scaled, y)
        y_pred = rf.predict(X_scaled)
        
        print(f"\n分类报告:")
        print(classification_report(y, y_pred))
        
        print(f"\n混淆矩阵:")
        cm = confusion_matrix(y, y_pred)
        print(cm)
        
        return cv_scores.mean(), cv_scores.std()
    
    def analyze_feature_correlation_matrix(self):
        """分析特征间相关性"""
        print("\n" + "="*60)
        print("特征间相关性分析")
        print("="*60)
        
        # 计算特征间相关系数
        feature_data = self.data[self.features].fillna(0)
        correlation_matrix = feature_data.corr()
        
        print("特征间相关系数矩阵:")
        print(correlation_matrix.round(4))
        
        # 找出高相关性的特征对
        print(f"\n高相关性特征对 (|r| > 0.7):")
        for i in range(len(self.features)):
            for j in range(i+1, len(self.features)):
                corr = correlation_matrix.iloc[i, j]
                if abs(corr) > 0.7:
                    print(f"  {self.features[i]} - {self.features[j]}: r = {corr:.4f}")
        
        return correlation_matrix
    
    def generate_summary_report(self):
        """生成总结报告"""
        print("\n" + "="*80)
        print("Delta特征分类可行性分析总结")
        print("="*80)
        
        # 数据概况
        print(f"1. 数据概况:")
        print(f"   - 总样本数: {len(self.data)}")
        print(f"   - 特征数: {len(self.features)}")
        print(f"   - 情绪类别数: {len(self.emotions)}")
        
        # 情绪分布
        emotion_counts = self.data['emotion'].value_counts()
        print(f"\n2. 情绪分布:")
        for emotion, count in emotion_counts.items():
            percentage = count / len(self.data) * 100
            print(f"   - {emotion}: {count} 条 ({percentage:.1f}%)")
        
        # 特征重要性
        feature_importance, feature_scores = self.analyze_feature_importance()
        print(f"\n3. 关键特征:")
        top_features = feature_importance.head(3)
        for _, row in top_features.iterrows():
            print(f"   - {row['feature']}: 重要性 {row['importance']:.4f}")
        
        # 分类性能
        accuracy, std = self.evaluate_classification_performance()
        print(f"\n4. 分类性能:")
        print(f"   - 交叉验证准确率: {accuracy:.4f} ± {std:.4f}")
        
        # 可行性评估
        print(f"\n5. 可行性评估:")
        if accuracy > 0.7:
            print(f"   ✅ 分类性能良好 (准确率 > 70%)")
        elif accuracy > 0.6:
            print(f"   ⚠️  分类性能一般 (准确率 60-70%)")
        else:
            print(f"   ❌ 分类性能较差 (准确率 < 60%)")
        
        # 建议
        print(f"\n6. 建议:")
        if accuracy > 0.7:
            print(f"   - Delta特征可以有效区分不同情绪")
            print(f"   - 建议使用随机森林等集成方法")
            print(f"   - 可考虑特征工程优化")
        elif accuracy > 0.6:
            print(f"   - Delta特征有一定区分能力")
            print(f"   - 建议增加更多特征或数据")
            print(f"   - 可尝试不同的分类算法")
        else:
            print(f"   - Delta特征区分能力有限")
            print(f"   - 建议重新考虑特征选择")
            print(f"   - 可能需要更多数据或特征工程")
    
    def run_analysis(self):
        """运行完整分析"""
        print("开始Delta特征与情绪标签相关性分析...")
        
        # 加载和清理数据
        if not self.load_data():
            return False
        
        if not self.clean_data():
            return False
        
        # 执行各项分析
        self.analyze_feature_distribution()
        self.analyze_emotion_feature_correlation()
        self.analyze_feature_correlation_matrix()
        self.generate_summary_report()
        
        return True

def main():
    """主函数"""
    analyzer = DeltaFeatureAnalyzer()
    success = analyzer.run_analysis()
    
    if success:
        print("\n分析完成!")
    else:
        print("分析失败!")

if __name__ == "__main__":
    main()
