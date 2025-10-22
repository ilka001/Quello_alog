#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
比较平均数和中位数差值结果
分析两种方法的差异和特点
"""

import csv
import statistics
from typing import Dict, List

def load_csv_data(filename: str) -> List[Dict]:
    """加载CSV数据"""
    data = []
    try:
        with open(filename, 'r', encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            for row in reader:
                data.append(row)
    except Exception as e:
        print(f"读取文件 {filename} 时出错: {e}")
    return data

def calculate_statistics(values: List[float]) -> Dict:
    """计算统计信息"""
    if not values:
        return {'mean': None, 'median': None, 'std': None, 'min': None, 'max': None, 'count': 0}
    
    return {
        'mean': statistics.mean(values),
        'median': statistics.median(values),
        'std': statistics.stdev(values) if len(values) > 1 else 0,
        'min': min(values),
        'max': max(values),
        'count': len(values)
    }

def compare_delta_results():
    """比较平均数和中位数差值结果"""
    print("HRV情绪差值分析：平均数 vs 中位数")
    print("="*60)
    
    # 加载数据
    mean_data = load_csv_data("hrv_emotion_delta_results.csv")
    median_data = load_csv_data("hrv_emotion_median_delta_results.csv")
    
    if not mean_data or not median_data:
        print("无法加载数据文件")
        return
    
    print(f"平均数差值数据: {len(mean_data)} 条记录")
    print(f"中位数差值数据: {len(median_data)} 条记录")
    
    # 特征列表
    features = ['delta_RMSSD', 'delta_pNN58', 'delta_SDNN', 'delta_SD1', 'delta_SD2', 'delta_SD1_SD2']
    emotions = ['悲伤', '愉悦', '焦虑']
    
    print("\n详细比较分析:")
    print("-" * 60)
    
    for emotion in emotions:
        print(f"\n{emotion} 情绪:")
        print("-" * 30)
        
        # 筛选对应情绪的数据
        mean_emotion_data = [row for row in mean_data if row['emotion'] == emotion]
        median_emotion_data = [row for row in median_data if row['emotion'] == emotion]
        
        for feature in features:
            print(f"\n{feature}:")
            
            # 提取特征值
            mean_values = []
            median_values = []
            
            for row in mean_emotion_data:
                if row.get(feature) and row[feature].strip():
                    try:
                        mean_values.append(float(row[feature]))
                    except (ValueError, TypeError):
                        continue
            
            for row in median_emotion_data:
                if row.get(feature) and row[feature].strip():
                    try:
                        median_values.append(float(row[feature]))
                    except (ValueError, TypeError):
                        continue
            
            # 计算统计信息
            mean_stats = calculate_statistics(mean_values)
            median_stats = calculate_statistics(median_values)
            
            print(f"  平均数差值 - 均值: {mean_stats['mean']:.4f}, 中位数: {mean_stats['median']:.4f}, 标准差: {mean_stats['std']:.4f}")
            print(f"  中位数差值 - 均值: {median_stats['mean']:.4f}, 中位数: {median_stats['median']:.4f}, 标准差: {median_stats['std']:.4f}")
            
            # 计算差异
            if mean_stats['mean'] is not None and median_stats['mean'] is not None:
                diff = abs(mean_stats['mean'] - median_stats['mean'])
                print(f"  均值差异: {diff:.4f}")
            
            if mean_stats['median'] is not None and median_stats['median'] is not None:
                diff = abs(mean_stats['median'] - median_stats['median'])
                print(f"  中位数差异: {diff:.4f}")
    
    # 整体分析
    print("\n整体分析:")
    print("-" * 30)
    
    # 计算所有特征的整体统计
    all_mean_values = []
    all_median_values = []
    
    for emotion in emotions:
        emotion_mean_data = [row for row in mean_data if row['emotion'] == emotion]
        emotion_median_data = [row for row in median_data if row['emotion'] == emotion]
        
        for feature in features:
            for row in emotion_mean_data:
                if row.get(feature) and row[feature].strip():
                    try:
                        all_mean_values.append(float(row[feature]))
                    except (ValueError, TypeError):
                        continue
            
            for row in emotion_median_data:
                if row.get(feature) and row[feature].strip():
                    try:
                        all_median_values.append(float(row[feature]))
                    except (ValueError, TypeError):
                        continue
    
    all_mean_stats = calculate_statistics(all_mean_values)
    all_median_stats = calculate_statistics(all_median_values)
    
    print(f"所有平均数差值的统计:")
    print(f"  均值: {all_mean_stats['mean']:.4f}")
    print(f"  中位数: {all_mean_stats['median']:.4f}")
    print(f"  标准差: {all_mean_stats['std']:.4f}")
    print(f"  范围: [{all_mean_stats['min']:.4f}, {all_mean_stats['max']:.4f}]")
    
    print(f"\n所有中位数差值的统计:")
    print(f"  均值: {all_median_stats['mean']:.4f}")
    print(f"  中位数: {all_median_stats['median']:.4f}")
    print(f"  标准差: {all_median_stats['std']:.4f}")
    print(f"  范围: [{all_median_stats['min']:.4f}, {all_median_stats['max']:.4f}]")
    
    # 计算相关性
    if len(all_mean_values) == len(all_median_values) and len(all_mean_values) > 0:
        # 简单的相关性计算
        n = len(all_mean_values)
        mean_mean = all_mean_stats['mean']
        median_mean = all_median_stats['mean']
        
        numerator = sum((all_mean_values[i] - mean_mean) * (all_median_values[i] - median_mean) 
                       for i in range(n))
        denominator = (sum((x - mean_mean) ** 2 for x in all_mean_values) * 
                      sum((x - median_mean) ** 2 for x in all_median_values)) ** 0.5
        
        if denominator != 0:
            correlation = numerator / denominator
            print(f"\n两种方法的相关性: {correlation:.4f}")
    
    print("\n分析完成!")

def main():
    """主函数"""
    compare_delta_results()

if __name__ == "__main__":
    main()
