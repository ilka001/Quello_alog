#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HRV特征提取器 - 10秒版本
从任意长度的CSV文件中提取HRV特征
输入: 包含时间和数值两列的CSV文件
输出: 一行包含6个HRV特征的数据

特点:
- 10秒版本，适合短时间数据分析
- 取消质量评分机制，直接处理所有数据
- 只要检测到峰值就计算特征，不论数据质量好坏

使用方法:
1. 修改代码顶部的配置参数（推荐）
2. 运行: python hrv_feature_extractor.py --use-config
3. 或者使用命令行参数: python hrv_feature_extractor.py input.csv -o output.csv -e "情绪标签"
"""

# ============================================================================================
# 配置参数 - 用户可修改的部分
# ============================================================================================

# 输入文件路径 - 用户可在此处直接修改要处理的CSV文件路径
INPUT_CSV_PATH = r"C:\Users\QAQ\Desktop\emotion\spe d\hrv_19700101_000656.csv"  # 修改为你的CSV文件路径，例如: "data/heart_rate.csv"

# 输出文件路径 - 用户可在此处直接修改输出文件路径（可选）
OUTPUT_CSV_PATH = "hrv_features.csv"  # 修改为你的输出文件路径，例如: "results/features.csv"

# 情绪标签 - 用户可在此处直接修改情绪标签
EMOTION_LABEL = "平静"  # 修改为你的情绪标签，例如: "愉悦"、"焦虑"、"悲伤"等

# 数据段长度（毫秒）- 用户可根据需要修改此参数
SEGMENT_DURATION_MS = 10000  # 10秒版本，可修改为5000(5秒)、15000(15秒)等

# 峰值检测参数 - 用户可根据信号质量调整
PEAK_DETECTION_PARAMS = {
    'distance': 5,           # 峰值间最小距离
    'prominence': 25,        # 峰值突出度
    'height': None           # 峰值高度阈值（None为自动）
}

# 质量评估参数 - 已禁用，直接处理所有数据
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

# ============================================================================================
# 导入依赖库
# ============================================================================================

import os
import sys
import numpy as np
import pandas as pd
from scipy.stats import zscore
from scipy.signal import find_peaks
import argparse
from typing import Tuple, Dict, Optional

class HRVFeatureExtractor:
    """HRV特征提取器类"""
    
    def __init__(self, segment_duration_ms: int = None):
        """
        初始化HRV特征提取器
        
        Args:
            segment_duration_ms: 数据段长度（毫秒），如果为None则使用全局配置
        """
        # 使用全局配置参数，如果传入了自定义参数则使用自定义参数
        self.segment_duration_ms = segment_duration_ms if segment_duration_ms is not None else SEGMENT_DURATION_MS
        
        # 使用全局峰值检测参数
        self.peak_detection_params = PEAK_DETECTION_PARAMS.copy()
        
        # 使用全局质量评估参数
        self.quality_params = QUALITY_PARAMS.copy()
    
    def detect_peaks(self, data: pd.DataFrame) -> Tuple[Optional[Dict], Optional[Dict]]:
        """
        对整段数据进行峰值检测
        
        Args:
            data: 包含'时间'和'数值'列的DataFrame
            
        Returns:
            (peak_info, properties): 峰值信息和检测属性
        """
        signal_values = data['数值'].values
        
        # 执行峰值检测
        peaks, properties = find_peaks(
            signal_values,
            distance=self.peak_detection_params['distance'],
            prominence=self.peak_detection_params['prominence'],
            height=self.peak_detection_params['height']
        )
        
        if len(peaks) == 0:
            return None, None
        
        # 计算峰值时间和RR间期
        peak_times = data['时间'].iloc[peaks].values
        rr_intervals = np.diff(peak_times)
        
        # 计算基本统计信息
        peak_info = {
            'indices': peaks,
            'times': peak_times,
            'values': signal_values[peaks],
            'rr_intervals': rr_intervals,
            'mean_rr': np.mean(rr_intervals),
            'std_rr': np.std(rr_intervals)
        }
        
        return peak_info, properties
    
    def detect_gaps_and_anomalies(self, peak_info: Dict) -> np.ndarray:
        """
        检测峰值间的断点和异常
        
        Args:
            peak_info: 峰值信息字典
            
        Returns:
            anomaly_indices: 异常位置索引
        """
        if peak_info is None or len(peak_info['rr_intervals']) == 0:
            return np.array([])
        
        rr_intervals = peak_info['rr_intervals']
        mean_rr = peak_info['mean_rr']
        
        # 方法1: 基于阈值因子的断点检测
        gap_threshold = mean_rr * self.quality_params['gap_threshold_factor']
        gap_indices = np.where(rr_intervals > gap_threshold)[0]
        
        # 方法2: 基于Z-score的异常检测
        outlier_indices = np.array([])
        try:
            if len(rr_intervals) > 1 and np.std(rr_intervals) > 1e-10:
                z_scores = np.abs(zscore(rr_intervals))
                valid_z_mask = np.isfinite(z_scores)
                if np.any(valid_z_mask):
                    outlier_indices = np.where((z_scores > self.quality_params['outlier_threshold']) & valid_z_mask)[0]
        except Exception:
            outlier_indices = np.array([])
        
        # 合并异常位置
        if len(gap_indices) > 0 and len(outlier_indices) > 0:
            anomaly_indices = np.unique(np.concatenate([gap_indices, outlier_indices]))
        elif len(gap_indices) > 0:
            anomaly_indices = gap_indices
        elif len(outlier_indices) > 0:
            anomaly_indices = outlier_indices
        else:
            anomaly_indices = np.array([])
        
        return anomaly_indices
    
    def evaluate_segment_quality(self, data_segment: pd.DataFrame, segment_peak_times: np.ndarray, 
                                segment_start_time: float, segment_end_time: float) -> Tuple[bool, Dict]:
        """
        评估单个数据段的质量
        
        Args:
            data_segment: 数据段
            segment_peak_times: 段内峰值时间
            segment_start_time: 段开始时间
            segment_end_time: 段结束时间
            
        Returns:
            (quality_passed, quality_info): 是否通过质量检查和质量信息
        """
        # 直接使用传入的峰值时间
        segment_peaks = segment_peak_times[(segment_peak_times >= segment_start_time) & (segment_peak_times <= segment_end_time)]
        
        quality_score = 0.0
        issues = []
        
        # 检查1: 峰值数量
        peak_count = len(segment_peaks)
        if peak_count < self.quality_params['min_peaks_per_segment']:
            return False, {
                'quality_score': 0.0,
                'peak_count': peak_count,
                'issues': [f"峰值数量不足({peak_count}<{self.quality_params['min_peaks_per_segment']})"],
                'rr_mean': 0,
                'rr_std': 0,
                'completeness': 0
            }
        elif peak_count > self.quality_params['max_peaks_per_segment']:
            return False, {
                'quality_score': 0.0,
                'peak_count': peak_count,
                'issues': [f"峰值数量过多({peak_count}>{self.quality_params['max_peaks_per_segment']})"],
                'rr_mean': 0,
                'rr_std': 0,
                'completeness': 0
            }
        else:
            quality_score += 0.3
        
        # 检查2: RR间期变异性
        if len(segment_peaks) > 1:
            rr_intervals = np.diff(segment_peaks)
            mean_rr = np.mean(rr_intervals)
            std_rr = np.std(rr_intervals)
            cv_rr = std_rr / mean_rr if mean_rr > 0 else float('inf')
            
            if cv_rr > self.quality_params['rr_variability_threshold']:
                return False, {
                    'quality_score': 0.0,
                    'peak_count': peak_count,
                    'issues': [f"RR间期变异性过高(CV={cv_rr:.3f})"],
                    'rr_mean': mean_rr,
                    'rr_std': std_rr,
                    'completeness': 0
                }
            else:
                quality_score += 0.3
            
            # 检查3: 断点检测
            gap_threshold = mean_rr * self.quality_params['gap_threshold_factor']
            gaps = np.sum(rr_intervals > gap_threshold)
            if gaps > 0:
                return False, {
                    'quality_score': 0.0,
                    'peak_count': peak_count,
                    'issues': [f"存在{gaps}个断点"],
                    'rr_mean': mean_rr,
                    'rr_std': std_rr,
                    'completeness': 0
                }
            else:
                quality_score += 0.2
            
            # 检查4: 异常值检测
            z_scores = np.abs(zscore(rr_intervals))
            outliers = np.sum(z_scores > self.quality_params['outlier_threshold'])
            if outliers > 0:
                return False, {
                    'quality_score': 0.0,
                    'peak_count': peak_count,
                    'issues': [f"存在{outliers}个异常值"],
                    'rr_mean': mean_rr,
                    'rr_std': std_rr,
                    'completeness': 0
                }
            else:
                quality_score += 0.2
            
            # 检查5: RR间隔范围检查
            rr_min_threshold = mean_rr * self.quality_params['rr_range_min_factor']
            rr_max_threshold = mean_rr * self.quality_params['rr_range_max_factor']
            out_of_range_count = np.sum((rr_intervals < rr_min_threshold) | (rr_intervals > rr_max_threshold))
            if out_of_range_count > 0:
                return False, {
                    'quality_score': 0.0,
                    'peak_count': peak_count,
                    'issues': [f"存在{out_of_range_count}个RR间隔超出70%-130%范围"],
                    'rr_mean': mean_rr,
                    'rr_std': std_rr,
                    'completeness': 0
                }
            else:
                quality_score += 0.2
        
        # 数据完整性检查
        expected_points = int(self.segment_duration_ms * len(data_segment) / (data_segment['时间'].max() - data_segment['时间'].min()))
        actual_points = len(data_segment)
        completeness = min(1.0, actual_points / expected_points)
        quality_score += 0.2 * completeness
        
        return True, {
            'quality_score': quality_score,
            'peak_count': peak_count,
            'issues': [],
            'rr_mean': np.mean(rr_intervals) if len(segment_peaks) > 1 else 0,
            'rr_std': np.std(rr_intervals) if len(segment_peaks) > 1 else 0,
            'completeness': completeness
        }
    
    def generate_candidate_segments(self, data: pd.DataFrame, peak_info: Dict) -> list:
        """
        生成候选数据段
        
        Args:
            data: 原始数据
            peak_info: 峰值信息
            
        Returns:
            candidates: 候选数据段列表
        """
        if peak_info is None or len(peak_info['times']) < 2:
            return []
        
        # 获取所有峰值时间
        peak_times = peak_info['times']
        data_start = data['时间'].min()
        data_end = data['时间'].max()
        
        candidates = []
        
        # 从每个峰值开始，尝试生成数据段
        for i, start_peak_time in enumerate(peak_times):
            # 计算窗口结束时间
            end_time = start_peak_time + self.segment_duration_ms
            
            # 检查是否超出数据范围
            if end_time > data_end:
                break
            
            # 提取数据段
            mask = (data['时间'] >= start_peak_time) & (data['时间'] <= end_time)
            segment_data = data[mask].copy().reset_index(drop=True)
            
            if len(segment_data) < 50:  # 10秒版本需要较少的数据点
                continue
            
            # 找到该段内的峰值时间
            segment_peak_times = peak_times[(peak_times >= start_peak_time) & (peak_times <= end_time)]
            
            # 评估质量
            quality_passed, quality = self.evaluate_segment_quality(segment_data, segment_peak_times, start_peak_time, end_time)
            
            # 只保留通过质量检查的段
            if quality_passed:
                candidates.append({
                    'start_time': start_peak_time,
                    'end_time': end_time,
                    'start_peak_index': i,
                    'data': segment_data,
                    'peak_times': segment_peak_times,
                    'quality': quality
                })
        
        # 按质量评分排序
        candidates.sort(key=lambda x: x['quality']['quality_score'], reverse=True)
        
        # 去重：如果两个候选段重叠度很高，只保留质量更高的
        filtered_candidates = []
        for candidate in candidates:
            is_duplicate = False
            for existing in filtered_candidates:
                # 计算重叠度
                overlap_start = max(candidate['start_time'], existing['start_time'])
                overlap_end = min(candidate['end_time'], existing['end_time'])
                overlap_duration = max(0, overlap_end - overlap_start)
                overlap_ratio = overlap_duration / self.segment_duration_ms
                
                # 如果重叠超过70%，认为是重复
                if overlap_ratio > 0.7:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                filtered_candidates.append(candidate)
        
        return filtered_candidates
    
    def calculate_hrv_features(self, rr_ms: np.ndarray) -> Dict:
        """
        计算HRV特征
        
        Args:
            rr_ms: RR间期数据（毫秒）
            
        Returns:
            features: HRV特征字典
        """
        if rr_ms is None or len(rr_ms) < 2:
            return {}
        
        feat = {}
        
        # 时域特征 - 向量化计算
        rr_diff = np.diff(rr_ms)
        
        # RMSSD - 向量化计算
        feat['RMSSD'] = float(np.sqrt(np.mean(rr_diff ** 2))) if len(rr_diff) > 0 else np.nan
        # feat['RMSSD'] = float(np.sqrt(np.mean(rr_diff ** 2))) if len(rr_diff) > 0 else np.nan
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
    
    def extract_features_from_csv(self, csv_path: str, emotion_label: str = "unknown") -> Optional[Dict]:
        """
        从CSV文件中提取HRV特征 - 简化版本，直接处理所有数据
        
        Args:
            csv_path: CSV文件路径
            emotion_label: 情绪标签
            
        Returns:
            result: 包含特征的结果字典，如果失败返回None
        """
        try:
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
            
            # 峰值检测
            peak_info, _ = self.detect_peaks(data)
            if peak_info is None or len(peak_info['times']) < 2:
                print(f"❌ 峰值检测失败或峰值不足: {os.path.basename(csv_path)}")
                return None
            
            # 直接使用所有峰值计算RR间期
            peak_times = peak_info['times']
            rr_intervals = np.diff(peak_times)
            
            # 计算HRV特征
            features = self.calculate_hrv_features(rr_intervals)
            
            if features:
                # 构建结果（只包含6个核心HRV特征）
                result = {
                    'RMSSD': features.get('RMSSD', np.nan),
                    'pNN58': features.get('pNN58', np.nan),
                    'SDNN': features.get('SDNN', np.nan),
                    'SD1': features.get('SD1', np.nan),
                    'SD2': features.get('SD2', np.nan),
                    'SD1_SD2': features.get('SD1_SD2', np.nan),
                    'emotion': emotion_label,
                    'peak_count': len(peak_times),
                    'segment_duration': self.segment_duration_ms / 1000.0
                }
                return result
            
            print(f"❌ HRV特征计算失败: {os.path.basename(csv_path)}")
            return None
            
        except Exception as e:
            print(f"❌ 处理失败: {os.path.basename(csv_path)} -> {e}")
            return None

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='从CSV文件中提取HRV特征')
    parser.add_argument('input_csv', nargs='?', default=INPUT_CSV_PATH, help=f'输入CSV文件路径（默认：{INPUT_CSV_PATH}）')
    parser.add_argument('-o', '--output', default=OUTPUT_CSV_PATH, help=f'输出CSV文件路径（默认：{OUTPUT_CSV_PATH}）')
    parser.add_argument('-e', '--emotion', default=EMOTION_LABEL, help=f'情绪标签（默认：{EMOTION_LABEL}）')
    parser.add_argument('-d', '--duration', type=int, default=SEGMENT_DURATION_MS, help=f'数据段长度（毫秒，默认：{SEGMENT_DURATION_MS}）')
    parser.add_argument('--use-config', action='store_true', help='使用代码顶部的配置参数，忽略命令行参数')
    
    args = parser.parse_args()
    
    # 如果使用配置模式，使用全局配置参数
    if args.use_config:
        input_csv = INPUT_CSV_PATH
        output_csv = OUTPUT_CSV_PATH
        emotion = EMOTION_LABEL
        duration = SEGMENT_DURATION_MS
        print("🔧 使用代码顶部的配置参数")
    else:
        input_csv = args.input_csv
        output_csv = args.output
        emotion = args.emotion
        duration = args.duration
        print("🔧 使用命令行参数")
    
    # 检查输入文件
    if not os.path.exists(input_csv):
        print(f"❌ 输入文件不存在: {input_csv}")
        print(f"💡 提示：请修改代码顶部的 INPUT_CSV_PATH 参数，或使用 --use-config 参数")
        sys.exit(1)
    
    print(f"🔍 开始处理文件: {input_csv}")
    print(f"🏷️ 情绪标签: {emotion}")
    print(f"⏱️ 数据段长度: {duration/1000:.1f} 秒")
    
    # 创建特征提取器
    extractor = HRVFeatureExtractor(segment_duration_ms=duration)
    
    # 提取特征
    result = extractor.extract_features_from_csv(input_csv, emotion)
    
    if result is None:
        print("❌ 特征提取失败")
        sys.exit(1)
    
    # 显示结果
    print("\n✅ 特征提取成功！")
    print(f"📊 提取的6个HRV特征:")
    feature_names = ['RMSSD', 'pNN58', 'SDNN', 'SD1', 'SD2', 'SD1_SD2']
    for name in feature_names:
        value = result[name]
        if np.isnan(value):
            print(f"   {name}: NaN")
        else:
            print(f"   {name}: {value:.4f}")
    
    print(f"\n📈 处理信息:")
    print(f"   峰值数量: {result['peak_count']}")
    print(f"   数据段长度: {result['segment_duration']:.1f} 秒")
    
    # 输出到文件
    if output_csv:
        # 创建DataFrame
        df = pd.DataFrame([result])
        
        # 按指定顺序排列列（只包含6个核心特征）
        cols = ['RMSSD', 'pNN58', 'SDNN', 'SD1', 'SD2', 'SD1_SD2']
        df = df[cols]
        
        # 保存到文件（不包含表头和索引）
        df.to_csv(output_csv, index=False, header=False, encoding='utf-8-sig')
        print(f"💾 结果已保存到: {output_csv}")
    
    print("\n🎉 处理完成！")

if __name__ == "__main__":
    main()
