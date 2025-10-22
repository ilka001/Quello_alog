#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快速特征测试
直接测试不同特征组合的分类效果
"""

import csv
import math
from collections import defaultdict
import statistics

def load_data():
    """加载数据"""
    data = []
    try:
        with open("hrv_emotion_delta_results.csv", 'r', encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get('person_name') and row.get('emotion'):
                    data.append(row)
        print(f"加载数据: {len(data)} 条记录")
        return data
    except Exception as e:
        print(f"加载失败: {e}")
        return []

def clean_data(data):
    """清理数据"""
    features = ['delta_RMSSD', 'delta_pNN58', 'delta_SDNN', 'delta_SD1', 'delta_SD2', 'delta_SD1_SD2']
    cleaned = []
    
    for row in data:
        valid_count = 0
        for feature in features:
            if row.get(feature) and row[feature].strip():
                try:
                    float(row[feature])
                    valid_count += 1
                except:
                    pass
        
        if valid_count >= 3:
            cleaned.append(row)
    
    print(f"清理后: {len(cleaned)} 条记录")
    return cleaned

def test_feature_combination(data, feature_names):
    """测试特征组合"""
    # 准备数据
    X = []
    y = []
    
    for row in data:
        values = []
        valid = True
        for feature in feature_names:
            if row.get(feature) and row[feature].strip():
                try:
                    values.append(float(row[feature]))
                except:
                    valid = False
                    break
            else:
                valid = False
                break
        
        if valid:
            X.append(values)
            y.append(row['emotion'])
    
    if len(X) < 20:
        return 0
    
    # 二分类：负面 vs 正面
    binary_X = []
    binary_y = []
    
    for i, emotion in enumerate(y):
        if emotion in ['悲伤', '焦虑']:
            binary_X.append(X[i])
            binary_y.append('negative')
        elif emotion == '愉悦':
            binary_X.append(X[i])
            binary_y.append('positive')
    
    if len(binary_X) < 10:
        return 0
    
    # 简单分类测试
    return simple_classify(binary_X, binary_y)

def simple_classify(X, y):
    """简单分类"""
    # 计算类别中心
    centers = defaultdict(list)
    for i, label in enumerate(y):
        centers[label].append(X[i])
    
    # 计算均值
    means = {}
    for label, data_list in centers.items():
        if data_list:
            if len(data_list[0]) == 1:
                means[label] = statistics.mean([x[0] for x in data_list])
            else:
                means[label] = [statistics.mean([x[j] for x in data_list]) for j in range(len(data_list[0]))]
    
    # 预测
    correct = 0
    for i, sample in enumerate(X):
        distances = {}
        for label, center in means.items():
            if isinstance(center, list):
                distance = math.sqrt(sum((sample[j] - center[j])**2 for j in range(len(sample))))
            else:
                distance = abs(sample[0] - center)
            distances[label] = distance
        
        predicted = min(distances.keys(), key=lambda x: distances[x])
        if predicted == y[i]:
            correct += 1
    
    return correct / len(X)

def main():
    """主函数"""
    print("快速特征测试开始...")
    
    # 加载数据
    data = load_data()
    if not data:
        return
    
    # 清理数据
    data = clean_data(data)
    
    # 测试不同特征组合
    features = ['delta_RMSSD', 'delta_pNN58', 'delta_SDNN', 'delta_SD1', 'delta_SD2', 'delta_SD1_SD2']
    
    print("\n单特征测试:")
    single_results = []
    for feature in features:
        accuracy = test_feature_combination(data, [feature])
        single_results.append((feature, accuracy))
        print(f"  {feature}: {accuracy:.4f}")
    
    # 排序
    single_results.sort(key=lambda x: x[1], reverse=True)
    
    print("\n单特征排名:")
    for i, (feature, acc) in enumerate(single_results):
        print(f"{i+1}. {feature}: {acc:.4f}")
    
    # 测试最佳特征对
    print("\n特征对测试:")
    pair_results = []
    for i in range(min(3, len(single_results))):
        for j in range(i+1, min(3, len(single_results))):
            feature1 = single_results[i][0]
            feature2 = single_results[j][0]
            accuracy = test_feature_combination(data, [feature1, feature2])
            pair_results.append(([feature1, feature2], accuracy))
            print(f"  {feature1} + {feature2}: {accuracy:.4f}")
    
    # 排序
    pair_results.sort(key=lambda x: x[1], reverse=True)
    
    print("\n特征对排名:")
    for i, (features, acc) in enumerate(pair_results):
        print(f"{i+1}. {' + '.join(features)}: {acc:.4f}")
    
    # 总结
    print("\n" + "="*50)
    print("总结")
    print("="*50)
    
    best_single = single_results[0]
    best_pair = pair_results[0] if pair_results else (None, 0)
    
    print(f"最佳单特征: {best_single[0]} ({best_single[1]:.4f})")
    if best_pair[0]:
        print(f"最佳特征对: {' + '.join(best_pair[0])} ({best_pair[1]:.4f})")
    
    if best_single[1] > 0.6:
        print("✅ 单特征效果良好")
    elif best_single[1] > 0.5:
        print("⚠️ 单特征效果一般")
    else:
        print("❌ 单特征效果较差")
    
    if best_pair[1] > best_single[1]:
        print("✅ 特征组合有改善")
    else:
        print("⚠️ 特征组合无改善")

if __name__ == "__main__":
    main()
