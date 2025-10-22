#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HRV情绪分类算法评估
测试多种机器学习算法在HRV情绪分类中的效果
"""

import csv
import math
import random
from collections import defaultdict, Counter
from typing import Dict, List, Tuple
import statistics

class HRVClassificationAlgorithms:
    def __init__(self, csv_file: str = "hrv_emotion_delta_results.csv"):
        self.csv_file = csv_file
        self.data = []
        self.features = ['delta_RMSSD', 'delta_pNN58', 'delta_SDNN', 'delta_SD1', 'delta_SD2', 'delta_SD1_SD2']
        
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
            for feature in self.features:
                if row.get(feature) and row[feature].strip():
                    try:
                        feature_values[feature] = float(row[feature])
                        valid_features += 1
                    except (ValueError, TypeError):
                        continue
            
            if valid_features >= 3:
                row['feature_values'] = feature_values
                cleaned_data.append(row)
        
        self.data = cleaned_data
        print(f"数据清理: {initial_count} -> {len(self.data)} 条记录")
        return True
    
    def prepare_data(self):
        """准备训练数据"""
        X = []
        y = []
        
        for row in self.data:
            feature_vector = []
            for feature in self.features:
                if feature in row['feature_values']:
                    feature_vector.append(row['feature_values'][feature])
                else:
                    feature_vector.append(0.0)
            
            X.append(feature_vector)
            y.append(row['emotion'])
        
        return X, y
    
    def normalize_features(self, X):
        """特征标准化"""
        if not X:
            return X
        
        # 计算每个特征的均值和标准差
        n_features = len(X[0])
        means = []
        stds = []
        
        for i in range(n_features):
            values = [row[i] for row in X]
            means.append(statistics.mean(values))
            stds.append(statistics.stdev(values) if len(values) > 1 else 1)
        
        # 标准化
        normalized_X = []
        for row in X:
            normalized_row = []
            for i in range(n_features):
                if stds[i] != 0:
                    normalized_row.append((row[i] - means[i]) / stds[i])
                else:
                    normalized_row.append(0.0)
            normalized_X.append(normalized_row)
        
        return normalized_X
    
    def cross_validate(self, X, y, classifier_func, n_folds=5):
        """交叉验证"""
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
            accuracy = classifier_func(X_train, y_train, X_test, y_test)
            fold_accuracies.append(accuracy)
        
        return statistics.mean(fold_accuracies), statistics.stdev(fold_accuracies) if len(fold_accuracies) > 1 else 0
    
    def knn_classifier(self, X_train, y_train, X_test, y_test, k=3):
        """K近邻分类器"""
        correct = 0
        
        for i, test_sample in enumerate(X_test):
            # 计算到所有训练样本的距离
            distances = []
            for j, train_sample in enumerate(X_train):
                distance = math.sqrt(sum((test_sample[k] - train_sample[k])**2 for k in range(len(test_sample))))
                distances.append((distance, y_train[j]))
            
            # 排序并选择最近的k个
            distances.sort(key=lambda x: x[0])
            k_nearest = distances[:k]
            
            # 投票决定类别
            votes = defaultdict(int)
            for _, label in k_nearest:
                votes[label] += 1
            
            predicted_label = max(votes.keys(), key=lambda x: votes[x])
            
            if predicted_label == y_test[i]:
                correct += 1
        
        return correct / len(y_test)
    
    def naive_bayes_classifier(self, X_train, y_train, X_test, y_test):
        """朴素贝叶斯分类器"""
        # 计算每个类别的先验概率
        class_counts = Counter(y_train)
        total_samples = len(y_train)
        prior_probs = {label: count/total_samples for label, count in class_counts.items()}
        
        # 计算每个特征在每个类别下的均值和标准差
        class_stats = defaultdict(lambda: defaultdict(list))
        for i, label in enumerate(y_train):
            for j, value in enumerate(X_train[i]):
                class_stats[label][j].append(value)
        
        # 计算均值和标准差
        class_means = {}
        class_stds = {}
        for label, features in class_stats.items():
            class_means[label] = {}
            class_stds[label] = {}
            for feature_idx, values in features.items():
                class_means[label][feature_idx] = statistics.mean(values)
                class_stds[label][feature_idx] = statistics.stdev(values) if len(values) > 1 else 1
        
        # 预测
        correct = 0
        for i, test_sample in enumerate(y_test):
            # 计算每个类别的后验概率
            posteriors = {}
            for label in class_counts.keys():
                # 先验概率
                posterior = math.log(prior_probs[label])
                
                # 似然概率
                for j, value in enumerate(X_test[i]):
                    mean = class_means[label][j]
                    std = class_stds[label][j]
                    
                    # 高斯概率密度函数
                    if std > 0:
                        likelihood = -0.5 * ((value - mean) / std) ** 2 - math.log(std * math.sqrt(2 * math.pi))
                        posterior += likelihood
                
                posteriors[label] = posterior
            
            # 选择概率最大的类别
            predicted_label = max(posteriors.keys(), key=lambda x: posteriors[x])
            
            if predicted_label == y_test[i]:
                correct += 1
        
        return correct / len(y_test)
    
    def decision_tree_classifier(self, X_train, y_train, X_test, y_test):
        """决策树分类器（简化版）"""
        # 构建简单的决策树
        tree = self.build_simple_tree(X_train, y_train)
        
        # 预测
        correct = 0
        for i, test_sample in enumerate(X_test):
            predicted_label = self.predict_tree(tree, test_sample)
            if predicted_label == y_test[i]:
                correct += 1
        
        return correct / len(y_test)
    
    def build_simple_tree(self, X, y, max_depth=3):
        """构建简单决策树"""
        if len(set(y)) == 1 or len(X) == 0:
            return {'label': y[0] if y else 'unknown'}
        
        if max_depth == 0:
            # 返回最常见的类别
            most_common = Counter(y).most_common(1)[0][0]
            return {'label': most_common}
        
        # 选择最佳分割特征
        best_feature, best_threshold = self.find_best_split(X, y)
        
        if best_feature is None:
            most_common = Counter(y).most_common(1)[0][0]
            return {'label': most_common}
        
        # 分割数据
        left_X, left_y, right_X, right_y = self.split_data(X, y, best_feature, best_threshold)
        
        # 递归构建子树
        left_tree = self.build_simple_tree(left_X, left_y, max_depth-1)
        right_tree = self.build_simple_tree(right_X, right_y, max_depth-1)
        
        return {
            'feature': best_feature,
            'threshold': best_threshold,
            'left': left_tree,
            'right': right_tree
        }
    
    def find_best_split(self, X, y):
        """寻找最佳分割点"""
        best_gini = float('inf')
        best_feature = None
        best_threshold = None
        
        for feature_idx in range(len(X[0])):
            values = [row[feature_idx] for row in X]
            unique_values = sorted(set(values))
            
            for i in range(len(unique_values)-1):
                threshold = (unique_values[i] + unique_values[i+1]) / 2
                gini = self.calculate_gini(X, y, feature_idx, threshold)
                
                if gini < best_gini:
                    best_gini = gini
                    best_feature = feature_idx
                    best_threshold = threshold
        
        return best_feature, best_threshold
    
    def calculate_gini(self, X, y, feature_idx, threshold):
        """计算基尼不纯度"""
        left_y = []
        right_y = []
        
        for i, row in enumerate(X):
            if row[feature_idx] <= threshold:
                left_y.append(y[i])
            else:
                right_y.append(y[i])
        
        if not left_y or not right_y:
            return float('inf')
        
        left_gini = 1 - sum((left_y.count(label) / len(left_y))**2 for label in set(left_y))
        right_gini = 1 - sum((right_y.count(label) / len(right_y))**2 for label in set(right_y))
        
        weighted_gini = (len(left_y) * left_gini + len(right_y) * right_gini) / len(y)
        return weighted_gini
    
    def split_data(self, X, y, feature_idx, threshold):
        """分割数据"""
        left_X, left_y = [], []
        right_X, right_y = [], []
        
        for i, row in enumerate(X):
            if row[feature_idx] <= threshold:
                left_X.append(row)
                left_y.append(y[i])
            else:
                right_X.append(row)
                right_y.append(y[i])
        
        return left_X, left_y, right_X, right_y
    
    def predict_tree(self, tree, sample):
        """使用决策树预测"""
        if 'label' in tree:
            return tree['label']
        
        if sample[tree['feature']] <= tree['threshold']:
            return self.predict_tree(tree['left'], sample)
        else:
            return self.predict_tree(tree['right'], sample)
    
    def svm_classifier(self, X_train, y_train, X_test, y_test):
        """支持向量机分类器（简化版）"""
        # 简化的SVM实现：使用线性核
        # 计算类别中心
        class_centers = defaultdict(list)
        for i, label in enumerate(y_train):
            class_centers[label].append(X_train[i])
        
        # 计算类别均值
        class_means = {}
        for label, data_list in class_centers.items():
            if data_list:
                means = []
                for j in range(len(data_list[0])):
                    values = [row[j] for row in data_list]
                    means.append(statistics.mean(values))
                class_means[label] = means
        
        # 计算类别间距离
        distances = {}
        emotions = list(class_means.keys())
        for i, emotion1 in enumerate(emotions):
            for emotion2 in emotions[i+1:]:
                distance = math.sqrt(sum((class_means[emotion1][j] - class_means[emotion2][j])**2 for j in range(len(class_means[emotion1]))))
                distances[(emotion1, emotion2)] = distance
        
        # 预测
        correct = 0
        for i, test_sample in enumerate(X_test):
            # 计算到每个类别中心的距离
            sample_distances = {}
            for label, center in class_means.items():
                distance = math.sqrt(sum((test_sample[j] - center[j])**2 for j in range(len(test_sample))))
                sample_distances[label] = distance
            
            # 选择距离最小的类别
            predicted_label = min(sample_distances.keys(), key=lambda x: sample_distances[x])
            
            if predicted_label == y_test[i]:
                correct += 1
        
        return correct / len(y_test)
    
    def random_forest_classifier(self, X_train, y_train, X_test, y_test, n_trees=10):
        """随机森林分类器（简化版）"""
        # 构建多个决策树
        trees = []
        for _ in range(n_trees):
            # 随机采样
            n_samples = len(X_train)
            sampled_indices = random.sample(range(n_samples), n_samples)
            sampled_X = [X_train[i] for i in sampled_indices]
            sampled_y = [y_train[i] for i in sampled_indices]
            
            # 构建树
            tree = self.build_simple_tree(sampled_X, sampled_y, max_depth=3)
            trees.append(tree)
        
        # 预测
        correct = 0
        for i, test_sample in enumerate(X_test):
            # 每个树投票
            votes = defaultdict(int)
            for tree in trees:
                prediction = self.predict_tree(tree, test_sample)
                votes[prediction] += 1
            
            # 选择得票最多的类别
            predicted_label = max(votes.keys(), key=lambda x: votes[x])
            
            if predicted_label == y_test[i]:
                correct += 1
        
        return correct / len(y_test)
    
    def evaluate_algorithms(self):
        """评估所有算法"""
        print("\n" + "="*80)
        print("HRV情绪分类算法评估")
        print("="*80)
        
        # 准备数据
        X, y = self.prepare_data()
        X_normalized = self.normalize_features(X)
        
        print(f"数据准备完成:")
        print(f"  样本数: {len(X)}")
        print(f"  特征数: {len(X[0])}")
        print(f"  类别数: {len(set(y))}")
        
        # 定义算法
        algorithms = [
            ("K近邻 (KNN)", lambda X_train, y_train, X_test, y_test: self.knn_classifier(X_train, y_train, X_test, y_test, k=3)),
            ("朴素贝叶斯", self.naive_bayes_classifier),
            ("决策树", self.decision_tree_classifier),
            ("支持向量机", self.svm_classifier),
            ("随机森林", self.random_forest_classifier),
        ]
        
        results = []
        
        for name, classifier_func in algorithms:
            print(f"\n测试算法: {name}")
            print("-" * 50)
            
            # 交叉验证
            accuracy, std = self.cross_validate(X_normalized, y, classifier_func)
            
            results.append({
                'algorithm': name,
                'accuracy': accuracy,
                'std': std
            })
            
            print(f"准确率: {accuracy:.4f} ± {std:.4f}")
        
        # 排序
        results.sort(key=lambda x: x['accuracy'], reverse=True)
        
        print(f"\n" + "="*80)
        print("算法性能排名")
        print("="*80)
        
        for i, result in enumerate(results):
            print(f"{i+1}. {result['algorithm']}: {result['accuracy']:.4f} ± {result['std']:.4f}")
        
        # 生成建议
        best_result = results[0]
        print(f"\n推荐算法: {best_result['algorithm']}")
        print(f"预期准确率: {best_result['accuracy']:.4f}")
        
        if best_result['accuracy'] > 0.7:
            print("✅ 推荐使用此算法")
        elif best_result['accuracy'] > 0.6:
            print("⚠️ 可以考虑使用此算法")
        else:
            print("❌ 不推荐使用此算法")
        
        return results
    
    def run_evaluation(self):
        """运行评估"""
        print("开始HRV情绪分类算法评估...")
        
        if not self.load_data():
            return False
        
        if not self.clean_data():
            return False
        
        # 评估算法
        results = self.evaluate_algorithms()
        
        return True

def main():
    evaluator = HRVClassificationAlgorithms()
    success = evaluator.run_evaluation()
    
    if success:
        print("\n算法评估完成!")
    else:
        print("评估失败!")

if __name__ == "__main__":
    main()
