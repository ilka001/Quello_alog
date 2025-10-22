#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
特征优化分析 - 修复版
尝试不同的特征组合和分类策略，寻找最佳的分类效果
"""

import csv
import math
import random
from collections import defaultdict, Counter
from typing import Dict, List, Tuple
import statistics

class FeatureOptimizer:
    def __init__(self, csv_file: str = "hrv_emotion_delta_results.csv"):
        self.csv_file = csv_file
        self.data = []
        self.all_features = ['delta_RMSSD', 'delta_pNN58', 'delta_SDNN', 'delta_SD1', 'delta_SD2', 'delta_SD1_SD2']
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
            feature_values = {}
            for feature in self.all_features:
                if row.get(feature) and row[feature].strip():
                    try:
                        feature_values[feature] = float(row[feature])
                        valid_features += 1
                    except (ValueError, TypeError):
                        continue
            
            if valid_features >= 3:  # 至少3个有效特征
                row['feature_values'] = feature_values
                cleaned_data.append(row)
        
        self.data = cleaned_data
        print(f"数据清理: {initial_count} -> {len(self.data)} 条记录")
        return True
    
    def analyze_feature_combinations(self):
        """分析不同特征组合的效果"""
        print("\n" + "="*80)
        print("特征组合分析")
        print("="*80)
        
        # 准备数据
        X_data = []
        y_data = []
        
        for row in self.data:
            feature_values = row['feature_values']
            emotion = row['emotion']
            
            # 构建特征向量
            feature_vector = []
            for feature in self.all_features:
                if feature in feature_values:
                    feature_vector.append(feature_values[feature])
                else:
                    feature_vector.append(0.0)  # 用0填充缺失值
            
            X_data.append(feature_vector)
            y_data.append(emotion)
        
        # 尝试不同的特征组合
        feature_combinations = [
            # 单特征
            [0], [1], [2], [3], [4], [5],  # 对应各个delta特征
            # 双特征组合
            [0, 1], [0, 2], [0, 3], [0, 4], [0, 5],
            [1, 2], [1, 3], [1, 4], [1, 5],
            [2, 3], [2, 4], [2, 5],
            [3, 4], [3, 5],
            [4, 5],
            # 三特征组合
            [0, 1, 2], [0, 1, 3], [0, 1, 4], [0, 1, 5],
            [0, 2, 3], [0, 2, 4], [0, 2, 5],
            [0, 3, 4], [0, 3, 5], [0, 4, 5],
            [1, 2, 3], [1, 2, 4], [1, 2, 5],
            [1, 3, 4], [1, 3, 5], [1, 4, 5],
            [2, 3, 4], [2, 3, 5], [2, 4, 5],
            [3, 4, 5],
            # 四特征组合
            [0, 1, 2, 3], [0, 1, 2, 4], [0, 1, 2, 5],
            [0, 1, 3, 4], [0, 1, 3, 5], [0, 1, 4, 5],
            [0, 2, 3, 4], [0, 2, 3, 5], [0, 2, 4, 5],
            [0, 3, 4, 5], [1, 2, 3, 4], [1, 2, 3, 5],
            [1, 2, 4, 5], [1, 3, 4, 5], [2, 3, 4, 5],
            # 五特征组合
            [0, 1, 2, 3, 4], [0, 1, 2, 3, 5], [0, 1, 2, 4, 5],
            [0, 1, 3, 4, 5], [0, 2, 3, 4, 5], [1, 2, 3, 4, 5],
            # 全部特征
            [0, 1, 2, 3, 4, 5]
        ]
        
        results = []
        
        for combo in feature_combinations:
            # 提取特征
            X_combo = [[x[i] for i in combo] for x in X_data]
            feature_names = [self.all_features[i] for i in combo]
            
            # 计算分离度
            separation_score = self.calculate_separation_score(X_combo, y_data)
            
            # 计算分类可行性
            feasibility_score = self.calculate_feasibility_score(X_combo, y_data)
            
            # 综合得分
            combined_score = (separation_score + feasibility_score) / 2
            
            results.append({
                'features': feature_names,
                'feature_indices': combo,
                'separation': separation_score,
                'feasibility': feasibility_score,
                'combined': combined_score
            })
        
        # 按综合得分排序
        results.sort(key=lambda x: x['combined'], reverse=True)
        
        print("特征组合效果排名 (前10名):")
        print("-" * 80)
        for i, result in enumerate(results[:10]):
            print(f"{i+1:2d}. {', '.join(result['features'])}")
            print(f"    分离度: {result['separation']:.4f}, 可行性: {result['feasibility']:.4f}, 综合: {result['combined']:.4f}")
        
        return results
    
    def calculate_separation_score(self, X, y):
        """计算特征分离度得分"""
        # 按情绪分组
        emotion_data = defaultdict(list)
        for i, emotion in enumerate(y):
            emotion_data[emotion].append(X[i])
        
        # 计算每个情绪的特征均值
        emotion_means = {}
        for emotion, data_list in emotion_data.items():
            if data_list:
                means = []
                for j in range(len(data_list[0])):  # 特征维度
                    values = [row[j] for row in data_list]
                    means.append(statistics.mean(values))
                emotion_means[emotion] = means
        
        # 计算情绪间距离
        if len(emotion_means) >= 2:
            distances = []
            emotions = list(emotion_means.keys())
            for i, emotion1 in enumerate(emotions):
                for emotion2 in emotions[i+1:]:
                    # 计算欧几里得距离
                    distance = 0
                    for j in range(len(emotion_means[emotion1])):
                        diff = emotion_means[emotion1][j] - emotion_means[emotion2][j]
                        distance += diff ** 2
                    distance = math.sqrt(distance)
                    distances.append(distance)
            
            if distances:
                return statistics.mean(distances)
        
        return 0
    
    def calculate_feasibility_score(self, X, y):
        """计算分类可行性得分"""
        # 计算特征间相关性
        if len(X) > 1 and len(X[0]) > 1:
            correlations = []
            for i in range(len(X[0])):
                for j in range(i+1, len(X[0])):
                    values_i = [row[i] for row in X]
                    values_j = [row[j] for row in X]
                    corr = self.pearson_correlation(values_i, values_j)
                    correlations.append(abs(corr))
            
            if correlations:
                avg_correlation = statistics.mean(correlations)
                # 相关性越低，可行性越高
                feasibility = 1.0 - avg_correlation
                return feasibility
        
        return 0.5
    
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
    
    def test_binary_classification(self, feature_indices, target_emotions):
        """测试二分类效果"""
        print(f"\n测试二分类: {target_emotions[0]} vs {target_emotions[1]}")
        print("-" * 50)
        
        # 准备数据
        X_data = []
        y_data = []
        
        for row in self.data:
            feature_values = row['feature_values']
            emotion = row['emotion']
            
            if emotion in target_emotions:
                # 构建特征向量
                feature_vector = []
                for i in feature_indices:
                    feature_name = self.all_features[i]
                    if feature_name in feature_values:
                        feature_vector.append(feature_values[feature_name])
                    else:
                        feature_vector.append(0.0)
                
                X_data.append(feature_vector)
                y_data.append(emotion)
        
        if len(X_data) < 10:
            print("样本数不足，跳过测试")
            return 0
        
        # 交叉验证
        accuracy = self.cross_validate_binary(X_data, y_data, target_emotions)
        return accuracy
    
    def cross_validate_binary(self, X, y, target_emotions, n_folds=5):
        """二分类交叉验证"""
        # 按类别分层
        class_data = defaultdict(list)
        for i, label in enumerate(y):
            class_data[label].append((X[i], label))
        
        # 计算每折的样本数
        fold_sizes = {}
        for class_name, data_list in class_data.items():
            fold_sizes[class_name] = len(data_list) // n_folds
        
        fold_accuracies = []
        
        for fold in range(n_folds):
            # 构建训练集和测试集
            train_data = []
            test_data = []
            
            for class_name, data_list in class_data.items():
                start_idx = fold * fold_sizes[class_name]
                end_idx = start_idx + fold_sizes[class_name]
                
                test_data.extend(data_list[start_idx:end_idx])
                train_data.extend(data_list[:start_idx] + data_list[end_idx:])
            
            # 准备数据
            X_train = [item[0] for item in train_data]
            y_train = [item[1] for item in train_data]
            X_test = [item[0] for item in test_data]
            y_test = [item[1] for item in test_data]
            
            # 训练和预测
            accuracy = self.train_and_predict_binary(X_train, y_train, X_test, y_test)
            fold_accuracies.append(accuracy)
        
        avg_accuracy = statistics.mean(fold_accuracies)
        print(f"交叉验证准确率: {avg_accuracy:.4f}")
        return avg_accuracy
    
    def train_and_predict_binary(self, X_train, y_train, X_test, y_test):
        """训练和预测二分类器"""
        # 计算每个类别的特征均值
        class_means = defaultdict(list)
        for i, label in enumerate(y_train):
            class_means[label].append(X_train[i])
        
        # 计算类别中心
        class_centers = {}
        for label, data_list in class_means.items():
            if data_list:
                center = []
                for j in range(len(data_list[0])):
                    values = [row[j] for row in data_list]
                    center.append(statistics.mean(values))
                class_centers[label] = center
        
        # 预测
        correct = 0
        for i, test_sample in enumerate(X_test):
            # 计算到每个类别中心的距离
            distances = {}
            for label, center in class_centers.items():
                distance = 0
                for j in range(len(test_sample)):
                    diff = test_sample[j] - center[j]
                    distance += diff ** 2
                distances[label] = math.sqrt(distance)
            
            # 选择距离最小的类别
            predicted_label = min(distances.keys(), key=lambda x: distances[x])
            
            if predicted_label == y_test[i]:
                correct += 1
        
        return correct / len(y_test)
    
    def find_best_combinations(self, results):
        """寻找最佳特征组合"""
        print("\n" + "="*80)
        print("最佳特征组合测试")
        print("="*80)
        
        # 选择前5个特征组合进行测试
        top_combinations = results[:5]
        
        # 测试不同的二分类策略
        classification_strategies = [
            (['悲伤', '焦虑'], ['愉悦']),  # 负面 vs 正面
            (['悲伤'], ['愉悦']),          # 悲伤 vs 愉悦
            (['焦虑'], ['愉悦']),          # 焦虑 vs 愉悦
            (['悲伤'], ['焦虑']),          # 悲伤 vs 焦虑
        ]
        
        best_results = []
        
        for combo in top_combinations:
            print(f"\n特征组合: {', '.join(combo['features'])}")
            print("-" * 50)
            
            combo_results = []
            for strategy in classification_strategies:
                accuracy = self.test_binary_classification(combo['feature_indices'], strategy)
                combo_results.append({
                    'strategy': strategy,
                    'accuracy': accuracy
                })
            
            # 找到最佳策略
            best_strategy = max(combo_results, key=lambda x: x['accuracy'])
            best_results.append({
                'features': combo['features'],
                'feature_indices': combo['feature_indices'],
                'best_strategy': best_strategy['strategy'],
                'accuracy': best_strategy['accuracy']
            })
        
        # 按准确率排序
        best_results.sort(key=lambda x: x['accuracy'], reverse=True)
        
        print(f"\n最佳特征组合排名:")
        print("-" * 80)
        for i, result in enumerate(best_results):
            print(f"{i+1}. 特征: {', '.join(result['features'])}")
            print(f"   策略: {result['best_strategy'][0]} vs {result['best_strategy'][1]}")
            print(f"   准确率: {result['accuracy']:.4f}")
        
        return best_results
    
    def generate_recommendations(self, best_results):
        """生成最终建议"""
        print("\n" + "="*80)
        print("最终建议")
        print("="*80)
        
        if best_results:
            best_result = best_results[0]
            accuracy = best_result['accuracy']
            
            print(f"推荐方案:")
            print(f"  特征组合: {', '.join(best_result['features'])}")
            print(f"  分类策略: {best_result['best_strategy'][0]} vs {best_result['best_strategy'][1]}")
            print(f"  预期准确率: {accuracy:.4f}")
            
            if accuracy > 0.7:
                print(f"\n✅ 推荐使用此方案")
                print(f"   - 分类性能良好")
                print(f"   - 可以投入实际应用")
            elif accuracy > 0.6:
                print(f"\n⚠️  可以考虑使用此方案")
                print(f"   - 分类性能一般")
                print(f"   - 需要进一步优化")
            else:
                print(f"\n❌ 不推荐使用此方案")
                print(f"   - 分类性能较差")
                print(f"   - 建议重新考虑特征选择")
            
            # 技术建议
            print(f"\n技术建议:")
            if len(best_result['features']) == 1:
                print(f"   - 使用简单的阈值分类器")
                print(f"   - 考虑特征标准化")
            elif len(best_result['features']) <= 3:
                print(f"   - 使用线性分类器 (逻辑回归、SVM)")
                print(f"   - 建议使用交叉验证")
            else:
                print(f"   - 使用集成方法 (随机森林、XGBoost)")
                print(f"   - 考虑特征选择")
            
            # 数据建议
            print(f"\n数据建议:")
            if accuracy < 0.6:
                print(f"   - 增加样本数量")
                print(f"   - 检查数据质量")
                print(f"   - 考虑特征工程")
            else:
                print(f"   - 当前数据质量可接受")
                print(f"   - 可以开始模型训练")
        
        else:
            print("未找到合适的特征组合")
    
    def run_optimization(self):
        """运行特征优化"""
        print("开始特征优化分析...")
        
        if not self.load_data():
            return False
        
        if not self.clean_data():
            return False
        
        # 分析特征组合
        results = self.analyze_feature_combinations()
        
        # 测试最佳组合
        best_results = self.find_best_combinations(results)
        
        # 生成建议
        self.generate_recommendations(best_results)
        
        return True

def main():
    optimizer = FeatureOptimizer()
    success = optimizer.run_optimization()
    
    if success:
        print("\n特征优化分析完成!")
    else:
        print("分析失败!")

if __name__ == "__main__":
    main()
