#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Delta特征分类模型测试
使用简单的机器学习算法测试Delta特征的分类效果
"""

import csv
import math
import random
from collections import defaultdict, Counter
from typing import Dict, List, Tuple
import statistics

class SimpleClassifier:
    """简单的分类器实现"""
    
    def __init__(self):
        self.feature_weights = {}
        self.emotion_centers = {}
        self.feature_means = {}
        self.feature_stds = {}
        
    def fit(self, X, y):
        """训练分类器"""
        # 计算特征标准化参数
        for i, feature in enumerate(['delta_RMSSD', 'delta_pNN58', 'delta_SDNN', 'delta_SD1', 'delta_SD2', 'delta_SD1_SD2']):
            values = [row[i] for row in X]
            self.feature_means[feature] = statistics.mean(values)
            self.feature_stds[feature] = statistics.stdev(values) if len(values) > 1 else 1
        
        # 计算每个情绪的特征中心
        emotion_data = defaultdict(list)
        for i, emotion in enumerate(y):
            emotion_data[emotion].append(X[i])
        
        for emotion, data_list in emotion_data.items():
            if data_list:
                center = []
                for j in range(len(data_list[0])):
                    values = [row[j] for row in data_list]
                    center.append(statistics.mean(values))
                self.emotion_centers[emotion] = center
        
        # 计算特征权重（基于分离度）
        self.calculate_feature_weights()
    
    def calculate_feature_weights(self):
        """计算特征权重"""
        features = ['delta_RMSSD', 'delta_pNN58', 'delta_SDNN', 'delta_SD1', 'delta_SD2', 'delta_SD1_SD2']
        
        for i, feature in enumerate(features):
            # 计算该特征在不同情绪间的分离度
            values_by_emotion = defaultdict(list)
            for emotion, center in self.emotion_centers.items():
                values_by_emotion[emotion].append(center[i])
            
            # 计算情绪间差异
            differences = []
            emotions = list(values_by_emotion.keys())
            for j, emotion1 in enumerate(emotions):
                for emotion2 in emotions[j+1:]:
                    diff = abs(values_by_emotion[emotion1][0] - values_by_emotion[emotion2][0])
                    differences.append(diff)
            
            if differences:
                self.feature_weights[feature] = statistics.mean(differences)
            else:
                self.feature_weights[feature] = 0
    
    def predict(self, X):
        """预测"""
        predictions = []
        for row in X:
            # 标准化特征
            normalized_row = []
            for i, feature in enumerate(['delta_RMSSD', 'delta_pNN58', 'delta_SDNN', 'delta_SD1', 'delta_SD2', 'delta_SD1_SD2']):
                normalized_value = (row[i] - self.feature_means[feature]) / self.feature_stds[feature]
                normalized_row.append(normalized_value)
            
            # 计算到每个情绪中心的加权距离
            distances = {}
            for emotion, center in self.emotion_centers.items():
                distance = 0
                for i, feature in enumerate(['delta_RMSSD', 'delta_pNN58', 'delta_SDNN', 'delta_SD1', 'delta_SD2', 'delta_SD1_SD2']):
                    diff = normalized_row[i] - center[i]
                    weight = self.feature_weights[feature]
                    distance += weight * (diff ** 2)
                distances[emotion] = math.sqrt(distance)
            
            # 选择距离最小的情绪
            predicted_emotion = min(distances.keys(), key=lambda x: distances[x])
            predictions.append(predicted_emotion)
        
        return predictions

class DeltaClassificationTest:
    def __init__(self, csv_file: str = "hrv_emotion_median_delta_results.csv"):
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
            feature_values = []
            for feature in self.features:
                if row.get(feature) and row[feature].strip():
                    try:
                        feature_values.append(float(row[feature]))
                        valid_features += 1
                    except (ValueError, TypeError):
                        feature_values.append(0.0)
                else:
                    feature_values.append(0.0)
            
            if valid_features >= 3:
                row['feature_values'] = feature_values
                cleaned_data.append(row)
        
        self.data = cleaned_data
        print(f"数据清理: {initial_count} -> {len(self.data)} 条记录")
        return True
    
    def split_data(self, test_ratio=0.2):
        """分割数据"""
        # 按情绪分层分割
        emotion_data = defaultdict(list)
        for row in self.data:
            emotion_data[row['emotion']].append(row)
        
        train_data = []
        test_data = []
        
        for emotion, data_list in emotion_data.items():
            # 随机打乱
            random.shuffle(data_list)
            
            # 分割
            split_idx = int(len(data_list) * (1 - test_ratio))
            train_data.extend(data_list[:split_idx])
            test_data.extend(data_list[split_idx:])
        
        # 再次打乱
        random.shuffle(train_data)
        random.shuffle(test_data)
        
        print(f"数据分割: 训练集 {len(train_data)} 条, 测试集 {len(test_data)} 条")
        
        return train_data, test_data
    
    def prepare_features_labels(self, data):
        """准备特征和标签"""
        X = []
        y = []
        
        for row in data:
            X.append(row['feature_values'])
            y.append(row['emotion'])
        
        return X, y
    
    def evaluate_classifier(self, y_true, y_pred):
        """评估分类器性能"""
        # 计算准确率
        correct = sum(1 for true, pred in zip(y_true, y_pred) if true == pred)
        accuracy = correct / len(y_true)
        
        # 计算每个情绪的精确率、召回率
        emotion_stats = {}
        for emotion in self.emotions:
            true_positives = sum(1 for true, pred in zip(y_true, y_pred) if true == emotion and pred == emotion)
            false_positives = sum(1 for true, pred in zip(y_true, y_pred) if true != emotion and pred == emotion)
            false_negatives = sum(1 for true, pred in zip(y_true, y_pred) if true == emotion and pred != emotion)
            
            precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
            recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            emotion_stats[emotion] = {
                'precision': precision,
                'recall': recall,
                'f1': f1
            }
        
        return accuracy, emotion_stats
    
    def cross_validation(self, n_folds=5):
        """交叉验证"""
        print(f"\n开始 {n_folds} 折交叉验证...")
        
        # 按情绪分层
        emotion_data = defaultdict(list)
        for row in self.data:
            emotion_data[row['emotion']].append(row)
        
        # 计算每折的样本数
        fold_sizes = {}
        for emotion, data_list in emotion_data.items():
            fold_sizes[emotion] = len(data_list) // n_folds
        
        fold_accuracies = []
        
        for fold in range(n_folds):
            print(f"\n第 {fold + 1} 折:")
            
            # 构建训练集和测试集
            train_data = []
            test_data = []
            
            for emotion, data_list in emotion_data.items():
                start_idx = fold * fold_sizes[emotion]
                end_idx = start_idx + fold_sizes[emotion]
                
                test_data.extend(data_list[start_idx:end_idx])
                train_data.extend(data_list[:start_idx] + data_list[end_idx:])
            
            # 准备数据
            X_train, y_train = self.prepare_features_labels(train_data)
            X_test, y_test = self.prepare_features_labels(test_data)
            
            # 训练和预测
            classifier = SimpleClassifier()
            classifier.fit(X_train, y_train)
            y_pred = classifier.predict(X_test)
            
            # 评估
            accuracy, emotion_stats = self.evaluate_classifier(y_test, y_pred)
            fold_accuracies.append(accuracy)
            
            print(f"  准确率: {accuracy:.4f}")
            for emotion, stats in emotion_stats.items():
                print(f"  {emotion}: 精确率={stats['precision']:.3f}, 召回率={stats['recall']:.3f}, F1={stats['f1']:.3f}")
        
        # 计算平均性能
        avg_accuracy = statistics.mean(fold_accuracies)
        std_accuracy = statistics.stdev(fold_accuracies) if len(fold_accuracies) > 1 else 0
        
        print(f"\n交叉验证结果:")
        print(f"  平均准确率: {avg_accuracy:.4f} ± {std_accuracy:.4f}")
        print(f"  各折准确率: {[f'{acc:.4f}' for acc in fold_accuracies]}")
        
        return avg_accuracy, std_accuracy
    
    def run_classification_test(self):
        """运行分类测试"""
        print("开始Delta特征分类测试...")
        
        if not self.load_data():
            return False
        
        if not self.clean_data():
            return False
        
        # 设置随机种子
        random.seed(42)
        
        # 交叉验证
        avg_accuracy, std_accuracy = self.cross_validation()
        
        # 评估结果
        print(f"\n" + "="*60)
        print("分类测试结果评估")
        print("="*60)
        
        print(f"交叉验证准确率: {avg_accuracy:.4f} ± {std_accuracy:.4f}")
        
        if avg_accuracy > 0.7:
            print("✅ 分类性能良好 (准确率 > 70%)")
            print("   Delta特征可以有效区分不同情绪")
        elif avg_accuracy > 0.6:
            print("⚠️  分类性能一般 (准确率 60-70%)")
            print("   Delta特征有一定区分能力，但需要优化")
        elif avg_accuracy > 0.5:
            print("⚠️  分类性能较差 (准确率 50-60%)")
            print("   Delta特征区分能力有限")
        else:
            print("❌ 分类性能很差 (准确率 < 50%)")
            print("   Delta特征无法有效区分情绪")
        
        # 与随机猜测对比
        random_accuracy = 1.0 / len(self.emotions)
        print(f"\n随机猜测准确率: {random_accuracy:.4f}")
        
        if avg_accuracy > random_accuracy * 1.5:
            print("✅ 显著优于随机猜测")
        elif avg_accuracy > random_accuracy * 1.2:
            print("⚠️  略优于随机猜测")
        else:
            print("❌ 与随机猜测相当")
        
        return True

def main():
    tester = DeltaClassificationTest()
    success = tester.run_classification_test()
    
    if success:
        print("\n分类测试完成!")
    else:
        print("分类测试失败!")

if __name__ == "__main__":
    main()
