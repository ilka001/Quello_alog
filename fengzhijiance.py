#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
峰值检测脚本
从CSV文件中检测前10秒的峰值并在终端输出可视化结果，计算6个HRV特征
使用与现有脚本完全相同的峰值检测和特征计算方法
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import zscore
from scipy.signal import find_peaks
import argparse
from typing import Tuple, Dict, Optional

# ============================================================================================
# 配置参数 - 与现有脚本保持一致
# ============================================================================================

# 默认输入文件路径
DEFAULT_INPUT_CSV = r"C:\Users\QAQ\Desktop\emotion\forMK\2.csv"

# 数据段长度（毫秒）- 默认10秒
SEGMENT_DURATION_MS = 10000

# 峰值检测参数 - 与现有脚本完全相同
PEAK_DETECTION_PARAMS = {
    'distance': 5,           # 峰值间最小距离
    'prominence': 25,        # 峰值突出度
    'height': None           # 峰值高度阈值（None为自动）
}

# 质量评估参数 - 与现有脚本完全相同
QUALITY_PARAMS = {
    'min_peaks_per_segment': 1,     # 最少峰值数（设为1，取消限制）
    'max_peaks_per_segment': 1000,  # 最多峰值数（设为很大值，取消限制）
    'gap_threshold_factor': 10.0,   # 断点检测阈值因子（设为很大值，取消限制）
    'rr_variability_threshold': 10.0, # RR间期变异性阈值（设为很大值，取消限制）
    'outlier_threshold': 10.0,      # 异常值检测阈值（设为很大值，取消限制）
    'rr_range_min_factor': 0.1,     # RR间隔范围下限因子（设为很小值，取消限制）
    'rr_range_max_factor': 10.0,    # RR间隔范围上限因子（设为很大值，取消限制）
    'min_segment_quality_score': -1 # 最低质量评分（设为负值，取消限制）
}

class PeakDetector:
    """峰值检测器类"""
    
    def __init__(self, segment_duration_ms: int = SEGMENT_DURATION_MS):
        """
        初始化峰值检测器
        
        Args:
            segment_duration_ms: 数据段长度（毫秒）
        """
        self.segment_duration_ms = segment_duration_ms
        self.peak_detection_params = PEAK_DETECTION_PARAMS.copy()
        self.quality_params = QUALITY_PARAMS.copy()
    
    def detect_peaks(self, data: pd.DataFrame) -> Tuple[Optional[Dict], Optional[Dict]]:
        """
        对整段数据进行峰值检测 - 与现有脚本完全相同
        
        Args:
            data: 包含'时间'和'数值'列的DataFrame
            
        Returns:
            (peak_info, properties): 峰值信息和检测属性
        """
        signal_values = data['数值'].values
        
        # 执行峰值检测 - 与现有脚本完全相同
        peaks, properties = find_peaks(
            signal_values,
            distance=self.peak_detection_params['distance'],
            prominence=self.peak_detection_params['prominence'],
            height=self.peak_detection_params['height']
        )
        
        if len(peaks) == 0:
            return None, None
        
        # 计算峰值时间和RR间期 - 与现有脚本完全相同
        peak_times = data['时间'].iloc[peaks].values
        rr_intervals = np.diff(peak_times)
        
        # 计算基本统计信息 - 与现有脚本完全相同
        peak_info = {
            'indices': peaks,
            'times': peak_times,
            'values': signal_values[peaks],
            'rr_intervals': rr_intervals,
            'mean_rr': np.mean(rr_intervals),
            'std_rr': np.std(rr_intervals)
        }
        
        return peak_info, properties
    
    def calculate_hrv_features(self, rr_ms: np.ndarray) -> Dict:
        """
        计算HRV特征 - 与现有脚本完全相同（不包括SampEn）
        
        Args:
            rr_ms: RR间期数据（毫秒）
            
        Returns:
            features: HRV特征字典
        """
        if rr_ms is None or len(rr_ms) < 2:
            return {}
        
        feat = {}
        
        # 时域特征 - 向量化计算 - 与现有脚本完全相同
        rr_diff = np.diff(rr_ms)
        
        # RMSSD - 向量化计算
        feat['RMSSD'] = float(np.sqrt(np.mean(rr_diff ** 2))) if len(rr_diff) > 0 else np.nan
        
        # pNN58 - 向量化计算
        if len(rr_diff) > 0:
            over_58_count = np.sum(np.abs(rr_diff) > 58.0)
            feat['pNN58'] = float((over_58_count / len(rr_diff)) * 100.0)
        else:
            feat['pNN58'] = np.nan
        
        # SDNN - 向量化计算
        feat['SDNN'] = float(np.std(rr_ms, ddof=1)) if len(rr_ms) > 1 else np.nan
        
        # Poincaré特征 - 向量化计算
        if len(rr_diff) > 0:
            var_diff = np.var(rr_diff)
            var_rr = np.var(rr_ms, ddof=1)
            
            sd1 = float(np.sqrt(var_diff / 2.0))
            sd2 = float(np.sqrt(2 * var_rr - var_diff / 2.0))
            
            feat['SD1'] = sd1
            feat['SD2'] = sd2
            feat['SD1_SD2'] = float(sd1 / sd2) if sd2 > 0 else np.nan
        else:
            feat['SD1'] = np.nan
            feat['SD2'] = np.nan
            feat['SD1_SD2'] = np.nan
        
        return feat
    
    def extract_first_10s_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        提取前10秒的数据
        
        Args:
            data: 原始数据
            
        Returns:
            first_10s_data: 前10秒的数据
        """
        start_time = data['时间'].min()
        end_time = start_time + self.segment_duration_ms
        
        # 提取前10秒数据
        mask = (data['时间'] >= start_time) & (data['时间'] <= end_time)
        first_10s_data = data[mask].copy().reset_index(drop=True)
        
        return first_10s_data
    
    def show_waveform_plot(self, data: pd.DataFrame, peak_info: Dict) -> None:
        """
        显示波形图窗口
        
        Args:
            data: 数据
            peak_info: 峰值信息
        """
        plt.figure(figsize=(12, 8))
        
        # 绘制原始数据波形
        plt.plot(data['时间'], data['数值'], 'b-', linewidth=1, alpha=0.7, label='原始信号')
        
        # 标记检测到的峰值
        if peak_info is not None and len(peak_info['times']) > 0:
            plt.plot(peak_info['times'], peak_info['values'], 'ro', markersize=8, 
                    label=f'检测到的峰值 (共{len(peak_info["times"])}个)')
        
        # 设置图形属性
        plt.xlabel('时间 (ms)', fontsize=12)
        plt.ylabel('信号值', fontsize=12)
        plt.title(f'HRV信号峰值检测结果 (前{self.segment_duration_ms/1000:.1f}秒)', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=11)
        
        # 添加统计信息
        if peak_info is not None and len(peak_info['times']) > 0:
            stats_text = f"峰值数量: {len(peak_info['times'])}\n"
            stats_text += f"平均RR间期: {peak_info['mean_rr']:.1f} ms\n"
            stats_text += f"RR间期标准差: {peak_info['std_rr']:.1f} ms"
            plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                    fontsize=10)
        
        plt.tight_layout()
        
        # 显示图片窗口
        print("🖼️ 显示波形图窗口...")
        plt.show()
    
    def process_csv_file(self, csv_path: str) -> Optional[Dict]:
        """
        处理CSV文件，检测峰值并计算HRV特征
        
        Args:
            csv_path: CSV文件路径
            
        Returns:
            result: 包含特征的结果字典，如果失败返回None
        """
        try:
            print(f"🔍 开始处理文件: {csv_path}")
            
            # 读取CSV文件
            data = pd.read_csv(csv_path, header=None)
            
            if data.shape[1] < 2:
                print(f"❌ 文件格式错误: {os.path.basename(csv_path)} (需要至少2列数据)")
                return None
            
            # 设置列名
            data = data.iloc[:, :2].copy()
            data.columns = ['时间', '数值']
            
            if len(data) < 10:
                print(f"❌ 数据不足: {os.path.basename(csv_path)} (需要至少10个数据点)")
                return None
            
            # 提取前10秒数据
            first_10s_data = self.extract_first_10s_data(data)
            print(f"📊 提取前{self.segment_duration_ms/1000:.1f}秒数据: {len(first_10s_data)} 个数据点")
            
            # 峰值检测
            peak_info, properties = self.detect_peaks(first_10s_data)
            if peak_info is None or len(peak_info['times']) < 2:
                print(f"❌ 峰值检测失败或峰值不足: {os.path.basename(csv_path)}")
                return None
            
            print(f"✅ 检测到 {len(peak_info['times'])} 个峰值")
            
            # 计算RR间期
            rr_intervals = peak_info['rr_intervals']
            print(f"📈 RR间期统计: 平均={np.mean(rr_intervals):.1f}ms, 标准差={np.std(rr_intervals):.1f}ms")
            
            # 显示波形图窗口
            self.show_waveform_plot(first_10s_data, peak_info)
            
            # 计算HRV特征
            features = self.calculate_hrv_features(rr_intervals)
            
            if features:
                # 构建结果
                result = {
                    'RMSSD': features.get('RMSSD', np.nan),
                    'pNN58': features.get('pNN58', np.nan),
                    'SDNN': features.get('SDNN', np.nan),
                    'SD1': features.get('SD1', np.nan),
                    'SD2': features.get('SD2', np.nan),
                    'SD1_SD2': features.get('SD1_SD2', np.nan),
                    'peak_count': len(peak_info['times']),
                    'segment_duration': self.segment_duration_ms / 1000.0,
                    'mean_rr': peak_info['mean_rr'],
                    'std_rr': peak_info['std_rr']
                }
                
                return result
            
            print(f"❌ HRV特征计算失败: {os.path.basename(csv_path)}")
            return None
            
        except Exception as e:
            print(f"❌ 处理失败: {os.path.basename(csv_path)} -> {e}")
            return None

def print_hrv_features(result: Dict) -> None:
    """
    在终端输出HRV特征结果
    
    Args:
        result: HRV特征结果字典
    """
    print("\n" + "="*60)
    print("🎯 HRV特征计算结果")
    print("="*60)
    
    # 输出6个核心HRV特征
    feature_names = ['RMSSD', 'pNN58', 'SDNN', 'SD1', 'SD2', 'SD1_SD2']
    feature_descriptions = {
        'RMSSD': '相邻RR间期差值的均方根',
        'pNN58': '相邻RR间期差值超过58ms的百分比',
        'SDNN': 'RR间期标准差',
        'SD1': 'Poincaré图的短轴标准差',
        'SD2': 'Poincaré图的长轴标准差',
        'SD1_SD2': 'SD1与SD2的比值'
    }
    
    for name in feature_names:
        value = result[name]
        description = feature_descriptions[name]
        if np.isnan(value):
            print(f"   {name:8}: NaN        ({description})")
        else:
            print(f"   {name:8}: {value:8.4f}  ({description})")
    
    print("\n" + "-"*60)
    print("📊 处理信息:")
    print(f"   峰值数量: {result['peak_count']}")
    print(f"   数据段长度: {result['segment_duration']:.1f} 秒")
    print(f"   平均RR间期: {result['mean_rr']:.1f} ms")
    print(f"   RR间期标准差: {result['std_rr']:.1f} ms")
    print("="*60)

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='峰值检测工具')
    parser.add_argument('input_csv', nargs='?', default=DEFAULT_INPUT_CSV, 
                       help=f'输入CSV文件路径（默认：{DEFAULT_INPUT_CSV}）')
    parser.add_argument('--duration', type=int, default=SEGMENT_DURATION_MS, 
                       help=f'数据段长度（毫秒，默认：{SEGMENT_DURATION_MS}）')
    
    args = parser.parse_args()
    
    # 检查输入文件
    if not os.path.exists(args.input_csv):
        print(f"❌ 输入文件不存在: {args.input_csv}")
        print(f"💡 提示：请修改代码顶部的 DEFAULT_INPUT_CSV 参数，或使用命令行参数")
        sys.exit(1)
    
    print(f"🔍 开始处理文件: {args.input_csv}")
    print(f"⏱️ 数据段长度: {args.duration/1000:.1f} 秒")
    
    # 创建检测器
    detector = PeakDetector(segment_duration_ms=args.duration)
    
    # 处理文件
    result = detector.process_csv_file(args.input_csv)
    
    if result is None:
        print("❌ 特征提取失败")
        sys.exit(1)
    
    # 在终端输出结果
    print_hrv_features(result)
    
    print("\n🎉 处理完成！")

if __name__ == "__main__":
    main()
