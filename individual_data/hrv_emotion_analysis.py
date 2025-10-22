#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HRV情绪分析脚本
分析不同人的HRV数据，计算各情绪与平静情绪的差值(Δ)
"""

import pandas as pd
import numpy as np
import os
import glob
from typing import Dict, List, Tuple
import argparse

class HRVEmotionAnalyzer:
    def __init__(self, data_dir: str = "data"):
        """
        初始化HRV情绪分析器
        
        Args:
            data_dir: 数据目录路径
        """
        self.data_dir = data_dir
        self.hrv_features = ['RMSSD', 'pNN58', 'SDNN', 'SD1', 'SD2', 'SD1_SD2']
        self.emotions = ['平静', '悲伤', '愉悦', '焦虑']
        
    def load_csv_files(self, selected_files: List[str] = None) -> Dict[str, pd.DataFrame]:
        """
        加载CSV文件
        
        Args:
            selected_files: 选择的文件列表，如果为None则加载所有文件
            
        Returns:
            文件名到DataFrame的映射
        """
        if selected_files is None:
            # 加载所有CSV文件
            csv_files = glob.glob(os.path.join(self.data_dir, "*.csv"))
        else:
            # 加载选择的文件
            csv_files = []
            for file in selected_files:
                file_path = os.path.join(self.data_dir, file)
                if os.path.exists(file_path):
                    csv_files.append(file_path)
                else:
                    print(f"警告: 文件 {file} 不存在")
        
        data_dict = {}
        for file_path in csv_files:
            filename = os.path.basename(file_path)
            try:
                df = pd.read_csv(file_path)
                data_dict[filename] = df
                print(f"已加载: {filename}")
            except Exception as e:
                print(f"错误: 无法加载文件 {filename}: {e}")
        
        return data_dict
    
    def extract_person_name(self, filename: str) -> str:
        """
        从文件名提取人名
        
        Args:
            filename: 文件名
            
        Returns:
            人名
        """
        # 移除文件扩展名
        name = filename.replace('_hrv_data.csv', '')
        # 移除开头的数字
        if name and name[0].isdigit():
            name = name[1:]
        return name
    
    def calculate_emotion_averages(self, df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """
        计算各情绪的特征平均值
        
        Args:
            df: 包含HRV数据的DataFrame
            
        Returns:
            情绪到特征平均值的映射
        """
        emotion_averages = {}
        
        for emotion in self.emotions:
            # 筛选当前情绪的数据
            emotion_data = df[df['emotion'].str.contains(emotion, na=False)]
            
            if len(emotion_data) > 0:
                emotion_averages[emotion] = {}
                for feature in self.hrv_features:
                    if feature in emotion_data.columns:
                        emotion_averages[emotion][feature] = emotion_data[feature].mean()
                    else:
                        print(f"警告: 特征 {feature} 在数据中不存在")
                        emotion_averages[emotion][feature] = np.nan
            else:
                print(f"警告: 未找到情绪 '{emotion}' 的数据")
                emotion_averages[emotion] = {feature: np.nan for feature in self.hrv_features}
        
        return emotion_averages
    
    def calculate_delta_values(self, emotion_averages: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, float]]:
        """
        计算其他情绪与平静情绪的差值(Δ)
        
        Args:
            emotion_averages: 各情绪的特征平均值
            
        Returns:
            情绪差值(Δ)字典
        """
        delta_values = {}
        
        # 获取平静情绪的平均值作为基准
        calm_averages = emotion_averages.get('平静', {})
        
        if not calm_averages or all(np.isnan(v) for v in calm_averages.values()):
            print("错误: 无法找到平静情绪的数据作为基准")
            return delta_values
        
        # 计算其他情绪与平静的差值
        for emotion in ['悲伤', '愉悦', '焦虑']:
            if emotion in emotion_averages:
                delta_values[emotion] = {}
                emotion_data = emotion_averages[emotion]
                
                for feature in self.hrv_features:
                    if feature in emotion_data and feature in calm_averages:
                        delta = emotion_data[feature] - calm_averages[feature]
                        delta_values[emotion][feature] = delta
                    else:
                        delta_values[emotion][feature] = np.nan
        
        return delta_values
    
    def analyze_person(self, filename: str, df: pd.DataFrame) -> Dict:
        """
        分析单个人的HRV数据
        
        Args:
            filename: 文件名
            df: 数据DataFrame
            
        Returns:
            分析结果字典
        """
        person_name = self.extract_person_name(filename)
        
        # 计算各情绪的平均值
        emotion_averages = self.calculate_emotion_averages(df)
        
        # 计算差值
        delta_values = self.calculate_delta_values(emotion_averages)
        
        result = {
            'person_name': person_name,
            'filename': filename,
            'emotion_averages': emotion_averages,
            'delta_values': delta_values
        }
        
        return result
    
    def analyze_multiple_people(self, selected_files: List[str] = None) -> List[Dict]:
        """
        分析多个人的HRV数据
        
        Args:
            selected_files: 选择的文件列表
            
        Returns:
            所有人的分析结果列表
        """
        # 加载数据
        data_dict = self.load_csv_files(selected_files)
        
        if not data_dict:
            print("错误: 没有找到有效的CSV文件")
            return []
        
        results = []
        for filename, df in data_dict.items():
            print(f"\n正在分析: {filename}")
            result = self.analyze_person(filename, df)
            results.append(result)
        
        return results
    
    def print_results(self, results: List[Dict]):
        """
        打印分析结果
        
        Args:
            results: 分析结果列表
        """
        print("\n" + "="*80)
        print("HRV情绪分析结果")
        print("="*80)
        
        for result in results:
            person_name = result['person_name']
            print(f"\n【{person_name}】")
            print("-" * 50)
            
            # 打印各情绪的平均值
            print("各情绪特征平均值:")
            emotion_averages = result['emotion_averages']
            for emotion in self.emotions:
                if emotion in emotion_averages:
                    print(f"  {emotion}:")
                    for feature in self.hrv_features:
                        if feature in emotion_averages[emotion]:
                            value = emotion_averages[emotion][feature]
                            if not np.isnan(value):
                                print(f"    {feature}: {value:.4f}")
                            else:
                                print(f"    {feature}: N/A")
            
            # 打印差值(Δ)
            print("\n与平静情绪的差值(Δ):")
            delta_values = result['delta_values']
            for emotion in ['悲伤', '愉悦', '焦虑']:
                if emotion in delta_values:
                    print(f"  {emotion}:")
                    for feature in self.hrv_features:
                        if feature in delta_values[emotion]:
                            delta = delta_values[emotion][feature]
                            if not np.isnan(delta):
                                print(f"    Δ{feature}: {delta:+.4f}")
                            else:
                                print(f"    Δ{feature}: N/A")
    
    def save_results_to_csv(self, results: List[Dict], output_file: str = "hrv_emotion_delta_results.csv"):
        """
        将结果保存到CSV文件
        
        Args:
            results: 分析结果列表
            output_file: 输出文件名
        """
        # 准备数据
        rows = []
        for result in results:
            person_name = result['person_name']
            delta_values = result['delta_values']
            
            for emotion in ['悲伤', '愉悦', '焦虑']:
                if emotion in delta_values:
                    row = {'person_name': person_name, 'emotion': emotion}
                    for feature in self.hrv_features:
                        if feature in delta_values[emotion]:
                            delta = delta_values[emotion][feature]
                            row[f'delta_{feature}'] = delta if not np.isnan(delta) else None
                        else:
                            row[f'delta_{feature}'] = None
                    rows.append(row)
        
        # 创建DataFrame并保存
        df_results = pd.DataFrame(rows)
        df_results.to_csv(output_file, index=False, encoding='utf-8-sig')
        print(f"\n结果已保存到: {output_file}")
    
    def list_available_files(self) -> List[str]:
        """
        列出可用的CSV文件
        
        Returns:
            文件名列表
        """
        csv_files = glob.glob(os.path.join(self.data_dir, "*.csv"))
        filenames = [os.path.basename(f) for f in csv_files]
        return sorted(filenames)

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='HRV情绪分析工具')
    parser.add_argument('--data-dir', default='data', help='数据目录路径')
    parser.add_argument('--files', nargs='*', help='指定要分析的文件名')
    parser.add_argument('--list-files', action='store_true', help='列出所有可用文件')
    parser.add_argument('--output', default='hrv_emotion_delta_results.csv', help='输出CSV文件名')
    
    args = parser.parse_args()
    
    # 创建分析器
    analyzer = HRVEmotionAnalyzer(args.data_dir)
    
    # 列出可用文件
    if args.list_files:
        files = analyzer.list_available_files()
        print("可用的CSV文件:")
        for i, file in enumerate(files, 1):
            person_name = analyzer.extract_person_name(file)
            print(f"{i:2d}. {file} ({person_name})")
        return
    
    # 交互式文件选择
    if args.files is None:
        files = analyzer.list_available_files()
        if not files:
            print("错误: 在指定目录中没有找到CSV文件")
            return
        
        print("可用的CSV文件:")
        for i, file in enumerate(files, 1):
            person_name = analyzer.extract_person_name(file)
            print(f"{i:2d}. {file} ({person_name})")
        
        print("\n请选择要分析的文件:")
        print("输入文件编号(用空格分隔，如: 1 3 5)，或输入 'all' 选择所有文件")
        
        choice = input("您的选择: ").strip()
        
        if choice.lower() == 'all':
            selected_files = files
        else:
            try:
                indices = [int(x) - 1 for x in choice.split()]
                selected_files = [files[i] for i in indices if 0 <= i < len(files)]
            except ValueError:
                print("错误: 请输入有效的数字")
                return
    else:
        selected_files = args.files
    
    if not selected_files:
        print("错误: 没有选择任何文件")
        return
    
    print(f"\n选择了 {len(selected_files)} 个文件进行分析...")
    
    # 执行分析
    results = analyzer.analyze_multiple_people(selected_files)
    
    if results:
        # 打印结果
        analyzer.print_results(results)
        
        # 保存结果
        analyzer.save_results_to_csv(results, args.output)
    else:
        print("错误: 没有成功分析任何文件")

if __name__ == "__main__":
    main()
