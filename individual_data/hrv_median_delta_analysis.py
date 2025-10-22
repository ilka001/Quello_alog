#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于中位数的HRV情绪差值分析
计算各情绪与平静情绪的中位数差值，生成CSV文件
"""

import pandas as pd
import numpy as np
import os
import glob
from typing import Dict, List
import warnings
warnings.filterwarnings('ignore')

class HRVMedianDeltaAnalyzer:
    def __init__(self, data_dir: str = "data"):
        """
        初始化HRV中位数差值分析器
        
        Args:
            data_dir: 数据目录路径
        """
        self.data_dir = data_dir
        self.hrv_features = ['RMSSD', 'pNN58', 'SDNN', 'SD1', 'SD2', 'SD1_SD2']
        self.emotions = ['平静', '悲伤', '愉悦', '焦虑']
        
    def extract_person_name(self, filename: str) -> str:
        """
        从文件名中提取人名
        
        Args:
            filename: 文件名
            
        Returns:
            提取的人名
        """
        # 移除路径和扩展名
        name = os.path.basename(filename).replace('_hrv_data.csv', '')
        
        # 移除开头的数字
        if name and name[0].isdigit():
            name = name[1:]
        return name
    
    def calculate_emotion_medians(self, df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """
        计算各情绪的特征中位数
        
        Args:
            df: 包含HRV数据的DataFrame
            
        Returns:
            情绪到特征中位数的映射
        """
        emotion_medians = {}
        
        for emotion in self.emotions:
            # 筛选当前情绪的数据
            emotion_data = df[df['emotion'].str.contains(emotion, na=False)]
            
            if len(emotion_data) > 0:
                emotion_medians[emotion] = {}
                for feature in self.hrv_features:
                    if feature in emotion_data.columns:
                        emotion_medians[emotion][feature] = emotion_data[feature].median()
                    else:
                        print(f"警告: 特征 {feature} 在数据中不存在")
                        emotion_medians[emotion][feature] = np.nan
            else:
                print(f"警告: 未找到情绪 '{emotion}' 的数据")
                emotion_medians[emotion] = {feature: np.nan for feature in self.hrv_features}
        
        return emotion_medians
    
    def calculate_delta_values(self, emotion_medians: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, float]]:
        """
        计算其他情绪与平静情绪的差值(Δ)
        
        Args:
            emotion_medians: 各情绪的特征中位数
            
        Returns:
            情绪差值(Δ)字典
        """
        delta_values = {}
        
        # 获取平静情绪的中位数作为基准
        calm_medians = emotion_medians.get('平静', {})
        
        if not calm_medians or all(np.isnan(v) for v in calm_medians.values()):
            print("错误: 无法找到平静情绪的数据作为基准")
            return delta_values
        
        # 计算其他情绪与平静的差值
        for emotion in ['悲伤', '愉悦', '焦虑']:
            if emotion in emotion_medians:
                delta_values[emotion] = {}
                emotion_data = emotion_medians[emotion]
                
                for feature in self.hrv_features:
                    if feature in emotion_data and feature in calm_medians:
                        delta = emotion_data[feature] - calm_medians[feature]
                        delta_values[emotion][feature] = delta
                    else:
                        delta_values[emotion][feature] = np.nan
        
        return delta_values
    
    def analyze_person(self, filename: str, df: pd.DataFrame) -> Dict:
        """
        分析单个人的HRV数据
        
        Args:
            filename: 文件名
            df: 包含HRV数据的DataFrame
            
        Returns:
            分析结果字典
        """
        person_name = self.extract_person_name(filename)
        
        # 计算各情绪的中位数
        emotion_medians = self.calculate_emotion_medians(df)
        
        # 计算差值
        delta_values = self.calculate_delta_values(emotion_medians)
        
        # 整理结果
        results = []
        for emotion in ['悲伤', '愉悦', '焦虑']:
            if emotion in delta_values:
                result = {
                    'person_name': person_name,
                    'emotion': emotion
                }
                
                # 添加差值特征
                for feature in self.hrv_features:
                    delta_key = f'delta_{feature}'
                    result[delta_key] = delta_values[emotion].get(feature, np.nan)
                
                results.append(result)
        
        return results
    
    def analyze_all_data(self) -> List[Dict]:
        """
        分析所有数据文件
        
        Returns:
            所有分析结果的列表
        """
        all_results = []
        
        # 获取所有CSV文件
        csv_files = glob.glob(os.path.join(self.data_dir, "*_hrv_data.csv"))
        
        if not csv_files:
            print(f"在目录 {self.data_dir} 中未找到HRV数据文件")
            return all_results
        
        print(f"找到 {len(csv_files)} 个数据文件")
        
        for file_path in csv_files:
            try:
                print(f"正在处理: {os.path.basename(file_path)}")
                
                # 读取数据
                df = pd.read_csv(file_path)
                
                # 分析数据
                person_results = self.analyze_person(file_path, df)
                all_results.extend(person_results)
                
            except Exception as e:
                print(f"处理文件 {file_path} 时出错: {e}")
                continue
        
        return all_results
    
    def save_results_to_csv(self, results: List[Dict], output_file: str = "hrv_emotion_median_delta_results.csv"):
        """
        将结果保存到CSV文件
        
        Args:
            results: 分析结果列表
            output_file: 输出文件名
        """
        if not results:
            print("没有结果可保存")
            return
        
        # 创建DataFrame
        df_results = pd.DataFrame(results)
        
        # 保存到CSV
        df_results.to_csv(output_file, index=False, encoding='utf-8-sig')
        print(f"结果已保存到: {output_file}")
        print(f"共保存 {len(results)} 条记录")
        
        # 显示统计信息
        print(f"\n统计信息:")
        print(f"人员数量: {df_results['person_name'].nunique()}")
        print(f"情绪类别: {df_results['emotion'].value_counts().to_dict()}")
        
        # 显示前几条记录
        print(f"\n前5条记录:")
        print(df_results.head().to_string(index=False))
    
    def run_analysis(self, output_file: str = "hrv_emotion_median_delta_results.csv"):
        """
        运行完整分析流程
        
        Args:
            output_file: 输出文件名
        """
        print("开始HRV中位数差值分析...")
        print("="*60)
        
        # 分析所有数据
        results = self.analyze_all_data()
        
        if results:
            # 保存结果
            self.save_results_to_csv(results, output_file)
            print("\n分析完成!")
        else:
            print("分析失败: 没有生成任何结果")

def main():
    """主函数"""
    analyzer = HRVMedianDeltaAnalyzer()
    analyzer.run_analysis()

if __name__ == "__main__":
    main()
