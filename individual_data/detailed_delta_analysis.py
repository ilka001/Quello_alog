#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
详细的Delta特征分析
深入分析HRV delta特征与情绪分类的相关性
"""

import csv
import math
from collections import defaultdict, Counter
from typing import Dict, List, Tuple
import statistics

class DetailedDeltaAnalyzer:
    def __init__(self, csv_file: str = "hrv_emotion_delta_results.csv"):
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
                    if row.get('person_name') and row.get('emotion'):
                        self.data.append(row)
            print(f"成功加载数据: {len(self.data)} 条记录")
            return True
        except Exception as e:
            print(f"加载数据失败: {e}")
            return False
    
    def clean_data(self):
        """清理数据"""
        initial_count = len(self.data)
        cleaned_data = []
        
        for row in self.data:
            valid_features = 0
            for feature in self.features:
                if row.get(feature) and row[feature].strip():
                    try:
                        float(row[feature])
                        valid_features += 1
                    except (ValueError, TypeError):
                        continue
            
            if valid_features >= 3:
                cleaned_data.append(row)
        
        self.data = cleaned_data
        print(f"数据清理: {initial_count} -> {len(self.data)} 条记录")
        return True
    
    def analyze_emotion_patterns(self):
        """分析情绪模式"""
        print("\n" + "="*80)
        print("情绪模式详细分析")
        print("="*80)
        
        # 按情绪分组数据
        emotion_data = defaultdict(list)
        for row in self.data:
            emotion = row['emotion']
            feature_values = {}
            for feature in self.features:
                if row.get(feature) and row[feature].strip():
                    try:
                        feature_values[feature] = float(row[feature])
                    except (ValueError, TypeError):
                        continue
            if len(feature_values) >= 3:  # 至少3个有效特征
                emotion_data[emotion].append(feature_values)
        
        # 计算每个情绪的特征统计
        emotion_stats = {}
        for emotion, data_list in emotion_data.items():
            if data_list:
                stats_dict = {}
                for feature in self.features:
                    values = [d[feature] for d in data_list if feature in d]
                    if values:
                        stats_dict[feature] = {
                            'mean': statistics.mean(values),
                            'std': statistics.stdev(values) if len(values) > 1 else 0,
                            'median': statistics.median(values),
                            'min': min(values),
                            'max': max(values),
                            'count': len(values)
                        }
                emotion_stats[emotion] = stats_dict
        
        # 打印详细统计
        for emotion in self.emotions:
            if emotion in emotion_stats:
                print(f"\n【{emotion}】情绪特征模式:")
                print("-" * 50)
                for feature in self.features:
                    if feature in emotion_stats[emotion]:
                        stats = emotion_stats[emotion][feature]
                        print(f"{feature:15s}: 均值={stats['mean']:+.4f}, 标准差={stats['std']:.4f}, "
                              f"中位数={stats['median']:+.4f}, 范围=[{stats['min']:+.2f}, {stats['max']:+.2f}]")
        
        return emotion_stats
    
    def calculate_separability_score(self, emotion_stats):
        """计算情绪分离度得分"""
        print("\n" + "="*80)
        print("情绪分离度分析")
        print("="*80)
        
        separability_scores = {}
        
        for feature in self.features:
            # 获取每个情绪的特征均值
            emotion_means = {}
            for emotion in self.emotions:
                if emotion in emotion_stats and feature in emotion_stats[emotion]:
                    emotion_means[emotion] = emotion_stats[emotion][feature]['mean']
            
            if len(emotion_means) >= 2:
                # 计算情绪间的最大差异
                values = list(emotion_means.values())
                max_diff = max(values) - min(values)
                
                # 计算情绪间的平均差异
                total_diff = 0
                count = 0
                for i, emotion1 in enumerate(self.emotions):
                    for emotion2 in self.emotions[i+1:]:
                        if emotion1 in emotion_means and emotion2 in emotion_means:
                            diff = abs(emotion_means[emotion1] - emotion_means[emotion2])
                            total_diff += diff
                            count += 1
                
                avg_diff = total_diff / count if count > 0 else 0
                
                # 计算分离度得分（结合最大差异和平均差异）
                separability_score = (max_diff + avg_diff) / 2
                separability_scores[feature] = separability_score
                
                print(f"{feature:15s}: 最大差异={max_diff:.4f}, 平均差异={avg_diff:.4f}, 分离度得分={separability_score:.4f}")
        
        # 排序
        sorted_scores = sorted(separability_scores.items(), key=lambda x: x[1], reverse=True)
        
        print(f"\n特征分离度排名:")
        for i, (feature, score) in enumerate(sorted_scores, 1):
            print(f"{i}. {feature}: {score:.4f}")
        
        return sorted_scores
    
    def analyze_feature_correlations(self):
        """分析特征间相关性"""
        print("\n" + "="*80)
        print("特征间相关性分析")
        print("="*80)
        
        # 构建特征矩阵
        feature_matrix = defaultdict(list)
        for row in self.data:
            for feature in self.features:
                if row.get(feature) and row[feature].strip():
                    try:
                        feature_matrix[feature].append(float(row[feature]))
                    except (ValueError, TypeError):
                        continue
        
        # 计算相关系数
        correlations = {}
        for i, feature1 in enumerate(self.features):
            for feature2 in self.features[i+1:]:
                if feature1 in feature_matrix and feature2 in feature_matrix:
                    values1 = feature_matrix[feature1]
                    values2 = feature_matrix[feature2]
                    
                    # 找到共同的有效值
                    common_values1 = []
                    common_values2 = []
                    for j in range(min(len(values1), len(values2))):
                        if j < len(values1) and j < len(values2):
                            common_values1.append(values1[j])
                            common_values2.append(values2[j])
                    
                    if len(common_values1) > 1:
                        # 计算皮尔逊相关系数
                        corr = self.pearson_correlation(common_values1, common_values2)
                        correlations[(feature1, feature2)] = corr
                        
                        print(f"{feature1:15s} - {feature2:15s}: r = {corr:+.4f}")
        
        # 找出高相关性的特征对
        high_corr_pairs = [(pair, corr) for pair, corr in correlations.items() if abs(corr) > 0.7]
        
        if high_corr_pairs:
            print(f"\n高相关性特征对 (|r| > 0.7):")
            for (feature1, feature2), corr in high_corr_pairs:
                print(f"  {feature1} - {feature2}: r = {corr:+.4f}")
        else:
            print(f"\n无高相关性特征对 (|r| > 0.7)")
        
        return correlations
    
    def pearson_correlation(self, x, y):
        """计算皮尔逊相关系数"""
        n = len(x)
        if n == 0:
            return 0
        
        sum_x = sum(x)
        sum_y = sum(y)
        sum_xy = sum(x[i] * y[i] for i in range(n))
        sum_x2 = sum(x[i] ** 2 for i in range(n))
        sum_y2 = sum(y[i] ** 2 for i in range(n))
        
        numerator = n * sum_xy - sum_x * sum_y
        denominator = math.sqrt((n * sum_x2 - sum_x ** 2) * (n * sum_y2 - sum_y ** 2))
        
        if denominator == 0:
            return 0
        
        return numerator / denominator
    
    def evaluate_classification_potential(self, emotion_stats, separability_scores):
        """评估分类潜力"""
        print("\n" + "="*80)
        print("分类潜力评估")
        print("="*80)
        
        # 1. 特征分离度评估
        avg_separability = statistics.mean([score for _, score in separability_scores])
        print(f"1. 特征分离度:")
        print(f"   平均分离度得分: {avg_separability:.4f}")
        
        if avg_separability > 5.0:
            print("   ✅ 分离度很高，分类潜力强")
        elif avg_separability > 3.0:
            print("   ⚠️  分离度中等，分类潜力一般")
        else:
            print("   ❌ 分离度较低，分类潜力弱")
        
        # 2. 情绪间差异评估
        print(f"\n2. 情绪间差异:")
        emotion_differences = {}
        for i, emotion1 in enumerate(self.emotions):
            for emotion2 in self.emotions[i+1:]:
                if emotion1 in emotion_stats and emotion2 in emotion_stats:
                    differences = []
                    for feature in self.features:
                        if (feature in emotion_stats[emotion1] and 
                            feature in emotion_stats[emotion2]):
                            diff = abs(emotion_stats[emotion1][feature]['mean'] - 
                                     emotion_stats[emotion2][feature]['mean'])
                            differences.append(diff)
                    
                    if differences:
                        avg_diff = statistics.mean(differences)
                        emotion_differences[f"{emotion1} vs {emotion2}"] = avg_diff
                        print(f"   {emotion1} vs {emotion2}: 平均差异 = {avg_diff:.4f}")
        
        # 3. 数据质量评估
        print(f"\n3. 数据质量:")
        emotion_counts = Counter(row['emotion'] for row in self.data)
        min_count = min(emotion_counts.values()) if emotion_counts else 0
        max_count = max(emotion_counts.values()) if emotion_counts else 0
        balance_ratio = min_count / max_count if max_count > 0 else 0
        
        print(f"   数据平衡性: {balance_ratio:.3f}")
        if balance_ratio > 0.8:
            print("   ✅ 数据平衡性很好")
        elif balance_ratio > 0.6:
            print("   ⚠️  数据平衡性一般")
        else:
            print("   ❌ 数据平衡性较差")
        
        # 4. 综合评估
        print(f"\n4. 综合评估:")
        
        # 计算综合得分
        separability_score = min(avg_separability / 5.0, 1.0)  # 归一化到0-1
        balance_score = balance_ratio
        difference_score = min(statistics.mean(emotion_differences.values()) / 3.0, 1.0) if emotion_differences else 0
        
        overall_score = (separability_score + balance_score + difference_score) / 3
        
        print(f"   综合得分: {overall_score:.3f}")
        
        if overall_score > 0.7:
            print("   ✅ 分类可行性很高")
            recommendation = "强烈推荐使用Delta特征进行情绪分类"
        elif overall_score > 0.5:
            print("   ⚠️  分类可行性中等")
            recommendation = "可以考虑使用Delta特征，但需要优化"
        else:
            print("   ❌ 分类可行性较低")
            recommendation = "不建议单独使用Delta特征进行分类"
        
        print(f"\n5. 建议:")
        print(f"   {recommendation}")
        
        if overall_score > 0.5:
            print("   - 推荐使用随机森林、SVM等算法")
            print("   - 可以考虑特征工程和特征选择")
            print("   - 建议使用交叉验证评估模型性能")
        else:
            print("   - 建议增加更多特征或数据")
            print("   - 考虑使用其他类型的特征")
            print("   - 可能需要更复杂的特征工程")
        
        return overall_score
    
    def run_analysis(self):
        """运行完整分析"""
        print("开始详细的Delta特征分析...")
        
        if not self.load_data():
            return False
        
        if not self.clean_data():
            return False
        
        # 执行各项分析
        emotion_stats = self.analyze_emotion_patterns()
        separability_scores = self.calculate_separability_score(emotion_stats)
        correlations = self.analyze_feature_correlations()
        overall_score = self.evaluate_classification_potential(emotion_stats, separability_scores)
        
        return True

def main():
    analyzer = DetailedDeltaAnalyzer()
    success = analyzer.run_analysis()
    
    if success:
        print("\n详细分析完成!")
    else:
        print("分析失败!")

if __name__ == "__main__":
    main()
