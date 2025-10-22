#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Delta特征与情绪标签相关性分析 - 简化版
分析HRV delta特征与情绪分类的相关性，验证分类模型的可行性
"""

import csv
import math
from collections import defaultdict, Counter
from typing import Dict, List, Tuple
import statistics

class DeltaFeatureAnalyzer:
    def __init__(self, csv_file: str = "hrv_emotion_delta_results.csv"):
        """
        初始化Delta特征分析器
        
        Args:
            csv_file: CSV文件路径
        """
        self.csv_file = csv_file
        self.data = []
        self.features = ['delta_RMSSD', 'delta_pNN58', 'delta_SDNN', 'delta_SD1', 'delta_SD2', 'delta_SD1_SD2']
        self.emotions = ['悲伤', '愉悦', '焦虑']
        
    def load_data(self):
        """加载数据"""
        try:
            with open(self.csv_file, 'r', encoding='utf-8-sig') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # 过滤掉空行
                    if row.get('person_name') and row.get('emotion'):
                        self.data.append(row)
            
            print(f"成功加载数据: {len(self.data)} 条记录")
            print(f"特征列: {self.features}")
            print(f"情绪类别: {self.emotions}")
            return True
        except Exception as e:
            print(f"加载数据失败: {e}")
            return False
    
    def clean_data(self):
        """清理数据"""
        if not self.data:
            return False
        
        # 移除空值行
        initial_count = len(self.data)
        cleaned_data = []
        
        for row in self.data:
            # 检查是否有足够的特征数据
            valid_features = 0
            for feature in self.features:
                if row.get(feature) and row[feature].strip():
                    try:
                        float(row[feature])
                        valid_features += 1
                    except (ValueError, TypeError):
                        continue
            
            # 至少要有3个有效特征才保留
            if valid_features >= 3:
                cleaned_data.append(row)
        
        self.data = cleaned_data
        final_count = len(self.data)
        
        print(f"数据清理: {initial_count} -> {final_count} 条记录")
        
        # 检查情绪分布
        emotion_counts = Counter(row['emotion'] for row in self.data)
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
            values = []
            for row in self.data:
                if row.get(feature) and row[feature].strip():
                    try:
                        values.append(float(row[feature]))
                    except (ValueError, TypeError):
                        continue
            
            if values:
                print(f"  均值: {statistics.mean(values):.4f}")
                print(f"  标准差: {statistics.stdev(values) if len(values) > 1 else 0:.4f}")
                print(f"  最小值: {min(values):.4f}")
                print(f"  最大值: {max(values):.4f}")
                print(f"  中位数: {statistics.median(values):.4f}")
                print(f"  样本数: {len(values)}")
            else:
                print("  无有效数据")
    
    def analyze_emotion_feature_correlation(self):
        """分析情绪与特征的相关性"""
        print("\n" + "="*60)
        print("情绪-特征相关性分析")
        print("="*60)
        
        # 为每个情绪计算特征统计
        emotion_stats = defaultdict(lambda: defaultdict(list))
        
        for row in self.data:
            emotion = row['emotion']
            for feature in self.features:
                if row.get(feature) and row[feature].strip():
                    try:
                        value = float(row[feature])
                        emotion_stats[emotion][feature].append(value)
                    except (ValueError, TypeError):
                        continue
        
        # 打印每个情绪的特征统计
        for emotion in self.emotions:
            if emotion in emotion_stats:
                print(f"\n{emotion} 情绪特征统计:")
                for feature in self.features:
                    if feature in emotion_stats[emotion] and emotion_stats[emotion][feature]:
                        values = emotion_stats[emotion][feature]
                        mean_val = statistics.mean(values)
                        std_val = statistics.stdev(values) if len(values) > 1 else 0
                        print(f"  {feature}: 均值={mean_val:+.4f}, 标准差={std_val:.4f}, 样本数={len(values)}")
        
        # 进行简单的方差分析
        print(f"\n特征区分能力分析:")
        for feature in self.features:
            all_values = []
            emotion_means = {}
            
            for emotion in self.emotions:
                if emotion in emotion_stats and feature in emotion_stats[emotion]:
                    values = emotion_stats[emotion][feature]
                    if values:
                        emotion_means[emotion] = statistics.mean(values)
                        all_values.extend(values)
            
            if len(emotion_means) >= 2 and all_values:
                # 计算组间方差和组内方差
                overall_mean = statistics.mean(all_values)
                
                # 组间方差
                between_group_var = 0
                for emotion, mean_val in emotion_means.items():
                    count = len(emotion_stats[emotion][feature])
                    between_group_var += count * (mean_val - overall_mean) ** 2
                
                # 组内方差
                within_group_var = 0
                for emotion in emotion_means:
                    values = emotion_stats[emotion][feature]
                    for value in values:
                        within_group_var += (value - emotion_means[emotion]) ** 2
                
                # F统计量
                if within_group_var > 0:
                    f_stat = between_group_var / within_group_var
                    print(f"  {feature}: F统计量={f_stat:.4f}")
                    if f_stat > 2.0:
                        print(f"    -> 有较好的区分能力")
                    elif f_stat > 1.0:
                        print(f"    -> 有一定的区分能力")
                    else:
                        print(f"    -> 区分能力较弱")
    
    def calculate_feature_importance(self):
        """计算特征重要性"""
        print("\n" + "="*60)
        print("特征重要性分析")
        print("="*60)
        
        # 使用简单的方差分析方法
        feature_scores = {}
        
        for feature in self.features:
            emotion_means = {}
            emotion_counts = {}
            
            # 计算每个情绪的特征均值
            for row in self.data:
                emotion = row['emotion']
                if row.get(feature) and row[feature].strip():
                    try:
                        value = float(row[feature])
                        if emotion not in emotion_means:
                            emotion_means[emotion] = 0
                            emotion_counts[emotion] = 0
                        emotion_means[emotion] += value
                        emotion_counts[emotion] += 1
                    except (ValueError, TypeError):
                        continue
            
            # 计算均值
            for emotion in emotion_means:
                if emotion_counts[emotion] > 0:
                    emotion_means[emotion] /= emotion_counts[emotion]
            
            # 计算特征重要性得分（基于均值差异）
            if len(emotion_means) >= 2:
                values = list(emotion_means.values())
                if values:
                    mean_diff = max(values) - min(values)
                    feature_scores[feature] = mean_diff
                else:
                    feature_scores[feature] = 0
            else:
                feature_scores[feature] = 0
        
        # 排序并显示
        sorted_features = sorted(feature_scores.items(), key=lambda x: x[1], reverse=True)
        
        print("特征重要性排序:")
        for i, (feature, score) in enumerate(sorted_features, 1):
            print(f"  {i}. {feature}: {score:.4f}")
        
        return sorted_features
    
    def evaluate_classification_feasibility(self):
        """评估分类可行性"""
        print("\n" + "="*60)
        print("分类可行性评估")
        print("="*60)
        
        # 计算每个情绪的特征模式
        emotion_patterns = {}
        
        for emotion in self.emotions:
            emotion_data = [row for row in self.data if row['emotion'] == emotion]
            if emotion_data:
                pattern = {}
                for feature in self.features:
                    values = []
                    for row in emotion_data:
                        if row.get(feature) and row[feature].strip():
                            try:
                                values.append(float(row[feature]))
                            except (ValueError, TypeError):
                                continue
                    
                    if values:
                        pattern[feature] = {
                            'mean': statistics.mean(values),
                            'std': statistics.stdev(values) if len(values) > 1 else 0,
                            'count': len(values)
                        }
                
                emotion_patterns[emotion] = pattern
        
        # 分析情绪间的特征差异
        print("情绪间特征差异分析:")
        for i, emotion1 in enumerate(self.emotions):
            for emotion2 in self.emotions[i+1:]:
                if emotion1 in emotion_patterns and emotion2 in emotion_patterns:
                    differences = []
                    for feature in self.features:
                        if (feature in emotion_patterns[emotion1] and 
                            feature in emotion_patterns[emotion2]):
                            diff = abs(emotion_patterns[emotion1][feature]['mean'] - 
                                     emotion_patterns[emotion2][feature]['mean'])
                            differences.append(diff)
                    
                    if differences:
                        avg_diff = statistics.mean(differences)
                        print(f"  {emotion1} vs {emotion2}: 平均差异 = {avg_diff:.4f}")
        
        # 计算整体分类可行性得分
        all_differences = []
        for emotion1 in self.emotions:
            for emotion2 in self.emotions:
                if emotion1 != emotion2 and emotion1 in emotion_patterns and emotion2 in emotion_patterns:
                    for feature in self.features:
                        if (feature in emotion_patterns[emotion1] and 
                            feature in emotion_patterns[emotion2]):
                            diff = abs(emotion_patterns[emotion1][feature]['mean'] - 
                                     emotion_patterns[emotion2][feature]['mean'])
                            all_differences.append(diff)
        
        if all_differences:
            overall_score = statistics.mean(all_differences)
            print(f"\n整体分类可行性得分: {overall_score:.4f}")
            
            if overall_score > 5.0:
                print("  ✅ 分类可行性很高")
            elif overall_score > 3.0:
                print("  ⚠️  分类可行性中等")
            else:
                print("  ❌ 分类可行性较低")
        
        return emotion_patterns
    
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
        emotion_counts = Counter(row['emotion'] for row in self.data)
        print(f"\n2. 情绪分布:")
        for emotion, count in emotion_counts.items():
            percentage = count / len(self.data) * 100
            print(f"   - {emotion}: {count} 条 ({percentage:.1f}%)")
        
        # 特征重要性
        feature_importance = self.calculate_feature_importance()
        print(f"\n3. 关键特征:")
        for i, (feature, score) in enumerate(feature_importance[:3], 1):
            print(f"   - {feature}: 重要性 {score:.4f}")
        
        # 分类可行性
        emotion_patterns = self.evaluate_classification_feasibility()
        
        # 可行性评估
        print(f"\n4. 可行性评估:")
        
        # 检查数据平衡性
        min_count = min(emotion_counts.values()) if emotion_counts else 0
        max_count = max(emotion_counts.values()) if emotion_counts else 0
        balance_ratio = min_count / max_count if max_count > 0 else 0
        
        if balance_ratio > 0.7:
            print(f"   ✅ 数据平衡性良好 (比例 > 0.7)")
        elif balance_ratio > 0.5:
            print(f"   ⚠️  数据平衡性一般 (比例 0.5-0.7)")
        else:
            print(f"   ❌ 数据平衡性较差 (比例 < 0.5)")
        
        # 检查特征区分能力
        feature_scores = [score for _, score in feature_importance]
        if feature_scores:
            avg_feature_score = statistics.mean(feature_scores)
            if avg_feature_score > 3.0:
                print(f"   ✅ 特征区分能力良好 (平均得分 > 3.0)")
            elif avg_feature_score > 1.0:
                print(f"   ⚠️  特征区分能力一般 (平均得分 1.0-3.0)")
            else:
                print(f"   ❌ 特征区分能力较弱 (平均得分 < 1.0)")
        
        # 建议
        print(f"\n5. 建议:")
        if balance_ratio > 0.7 and feature_scores and statistics.mean(feature_scores) > 3.0:
            print(f"   - Delta特征可以有效区分不同情绪")
            print(f"   - 建议使用随机森林、SVM等分类算法")
            print(f"   - 可考虑特征工程优化")
        elif balance_ratio > 0.5 and feature_scores and statistics.mean(feature_scores) > 1.0:
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
