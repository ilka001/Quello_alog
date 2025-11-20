#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多线程HRV特征计算脚本 - 40秒版本
合并了智能数据段处理和HRV特征计算功能
支持多线程并行处理CSV文件
使用40秒滑动窗口检测高质量数据段
"""

import os
import math
import glob
import numpy as np
import pandas as pd
from scipy import signal
from scipy.stats import entropy, zscore
from scipy.signal import find_peaks
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Tuple, Dict
import time
import sys
import subprocess

# --------------------------------------------------------------------------------------------
# 配置区域 - 用户需要修改的部分
# --------------------------------------------------------------------------------------------

# 输入数据目录（包含已划分好的CSV文件）
INPUT_DATA_DIR = r"C:\Users\QAQ\Desktop\emotion\1111"

# 输出文件
OUTPUT_FILE = r"./hrv_data.csv"

# RR间隔数据备份目录（每个标签单独保存）
RR_BACKUP_BASE_DIR = r"./backup"  # 40秒版本备份目录

# 最大线程数（建议设置为CPU核心数的1-2倍，因为HRV计算是CPU密集型）
MAX_WORKERS = 14

# 智能数据段处理参数 - 40秒版本
SEGMENT_DURATION_MS = 10000  # 数据段长度（毫秒）- 40秒
PEAK_DETECTION_PARAMS = {
    'distance': 5,           # 峰值间最小距离
    'prominence': 25,        # 峰值突出度
    'height': None           # 峰值高度阈值（None为自动）
}

# 质量评估参数 - 针对40秒数据段调整
QUALITY_PARAMS = {
    'min_peaks_per_segment': 6,     # 每段最少峰值数（40秒需要更多峰值）
    'max_peaks_per_segment': 20,     # 每段最多峰值数（40秒可以容纳更多峰值）
    'gap_threshold_factor': 2.5,     # 断点检测阈值因子（倍数）
    'rr_variability_threshold': 0.3, # RR间期变异性阈值
    'outlier_threshold': 2.5,        # 异常值检测阈值（Z-score）
    'rr_range_min_factor': 0.7,      # RR间隔范围下限因子（70%）
    'rr_range_max_factor': 1.3,      # RR间隔范围上限因子（130%）
    'min_segment_quality_score': 1.1 # 最低质量评分
}

# --------------------------------------------------------------------------------------------
# 智能数据段处理核心函数
# --------------------------------------------------------------------------------------------

def detect_all_peaks(data: pd.DataFrame) -> tuple:
    """对整段数据进行峰值检测"""
    signal_values = data['数值'].values
    
    # 执行峰值检测
    peaks, properties = find_peaks(
        signal_values,
        distance=PEAK_DETECTION_PARAMS['distance'],
        prominence=PEAK_DETECTION_PARAMS['prominence'],
        height=PEAK_DETECTION_PARAMS['height']
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

def detect_gaps_and_anomalies(peak_info: dict) -> np.ndarray:
    """检测峰值间的断点和异常"""
    if peak_info is None or len(peak_info['rr_intervals']) == 0:
        return np.array([])
    
    rr_intervals = peak_info['rr_intervals']
    mean_rr = peak_info['mean_rr']
    
    # 方法1: 基于阈值因子的断点检测
    gap_threshold = mean_rr * QUALITY_PARAMS['gap_threshold_factor']
    gap_indices = np.where(rr_intervals > gap_threshold)[0]
    
    # 方法2: 基于Z-score的异常检测
    outlier_indices = np.array([])
    try:
        if len(rr_intervals) > 1 and np.std(rr_intervals) > 1e-10:
            z_scores = np.abs(zscore(rr_intervals))
            valid_z_mask = np.isfinite(z_scores)
            if np.any(valid_z_mask):
                outlier_indices = np.where((z_scores > QUALITY_PARAMS['outlier_threshold']) & valid_z_mask)[0]
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

def evaluate_segment_quality(data_segment: pd.DataFrame, segment_peak_times: np.ndarray, 
                           segment_start_time: float, segment_end_time: float) -> tuple:
    """评估单个40s数据段的质量，返回(是否通过, 质量信息)"""
    
    # 直接使用传入的峰值时间
    segment_peaks = segment_peak_times[(segment_peak_times >= segment_start_time) & (segment_peak_times <= segment_end_time)]
    
    quality_score = 0.0
    issues = []
    
    # 检查1: 峰值数量 - 直接剔除不符合条件的数据段
    peak_count = len(segment_peaks)
    if peak_count < QUALITY_PARAMS['min_peaks_per_segment']:
        return False, {
            'quality_score': 0.0,
            'peak_count': peak_count,
            'issues': [f"峰值数量不足({peak_count}<{QUALITY_PARAMS['min_peaks_per_segment']})"],
            'rr_mean': 0,
            'rr_std': 0,
            'completeness': 0
        }
    elif peak_count > QUALITY_PARAMS['max_peaks_per_segment']:
        return False, {
            'quality_score': 0.0,
            'peak_count': peak_count,
            'issues': [f"峰值数量过多({peak_count}>{QUALITY_PARAMS['max_peaks_per_segment']})"],
            'rr_mean': 0,
            'rr_std': 0,
            'completeness': 0
        }
    else:
        quality_score += 0.3  # 峰值数量合理
    
    # 检查2: RR间期变异性 - 直接剔除变异性过高的数据段
    if len(segment_peaks) > 1:
        rr_intervals = np.diff(segment_peaks)
        mean_rr = np.mean(rr_intervals)
        std_rr = np.std(rr_intervals)
        cv_rr = std_rr / mean_rr if mean_rr > 0 else float('inf')
        
        if cv_rr > QUALITY_PARAMS['rr_variability_threshold']:
            return False, {
                'quality_score': 0.0,
                'peak_count': peak_count,
                'issues': [f"RR间期变异性过高(CV={cv_rr:.3f})"],
                'rr_mean': mean_rr,
                'rr_std': std_rr,
                'completeness': 0
            }
        else:
            quality_score += 0.3  # 变异性合理
        
        # 检查3: 断点检测 - 直接剔除存在断点的数据段
        gap_threshold = mean_rr * QUALITY_PARAMS['gap_threshold_factor']
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
            quality_score += 0.2  # 无断点
        
        # 检查4: 异常值检测 - 直接剔除存在异常值的数据段
        z_scores = np.abs(zscore(rr_intervals))
        outliers = np.sum(z_scores > QUALITY_PARAMS['outlier_threshold'])
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
            quality_score += 0.2  # 无异常值
        
        # 检查5: RR间隔范围检查（70%-130%）- 直接剔除超出范围的数据段
        rr_min_threshold = mean_rr * QUALITY_PARAMS['rr_range_min_factor']
        rr_max_threshold = mean_rr * QUALITY_PARAMS['rr_range_max_factor']
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
            quality_score += 0.2  # 所有RR间隔在合理范围内
    
    # 数据完整性检查
    expected_points = int(SEGMENT_DURATION_MS * len(data_segment) / (data_segment['时间'].max() - data_segment['时间'].min()))
    actual_points = len(data_segment)
    completeness = min(1.0, actual_points / expected_points)
    quality_score += 0.2 * completeness
    
    # 如果所有检查都通过，返回通过状态
    return True, {
        'quality_score': quality_score,
        'peak_count': peak_count,
        'issues': [],
        'rr_mean': np.mean(rr_intervals) if len(segment_peaks) > 1 else 0,
        'rr_std': np.std(rr_intervals) if len(segment_peaks) > 1 else 0,
        'completeness': completeness
    }

def generate_candidate_segments(data: pd.DataFrame, peak_info: dict) -> list:
    """生成候选的40s数据段 - 基于峰值位置的滑动窗口"""
    
    if peak_info is None or len(peak_info['times']) < 2:
        return []
    
    # 检测异常位置
    anomaly_indices = detect_gaps_and_anomalies(peak_info)
    
    # 获取所有峰值时间
    peak_times = peak_info['times']
    data_start = data['时间'].min()
    data_end = data['时间'].max()
    
    candidates = []
    
    # 从每个峰值开始，尝试生成40s的数据段
    for i, start_peak_time in enumerate(peak_times):
        # 计算窗口结束时间
        end_time = start_peak_time + SEGMENT_DURATION_MS
        
        # 检查是否超出数据范围
        if end_time > data_end:
            break
        
        # 提取数据段
        mask = (data['时间'] >= start_peak_time) & (data['时间'] <= end_time)
        segment_data = data[mask].copy().reset_index(drop=True)
        
        if len(segment_data) < 200:  # 40秒需要更多数据点
            continue
        
        # 找到该段内的峰值时间
        segment_peak_times = peak_times[(peak_times >= start_peak_time) & (peak_times <= end_time)]
        
        # 评估质量 - 直接剔除有问题的数据段
        quality_passed, quality = evaluate_segment_quality(segment_data, segment_peak_times, start_peak_time, end_time)
        
        # 只保留通过质量检查的段
        if quality_passed:
            candidates.append({
                'start_time': start_peak_time,
                'end_time': end_time,
                'start_peak_index': i,  # 记录起始峰值索引
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
            overlap_ratio = overlap_duration / SEGMENT_DURATION_MS
            
            # 如果重叠超过70%，认为是重复
            if overlap_ratio > 0.7:
                is_duplicate = True
                break
        
        if not is_duplicate:
            filtered_candidates.append(candidate)
    
    return filtered_candidates

# --------------------------------------------------------------------------------------------
# HRV特征计算核心函数
# --------------------------------------------------------------------------------------------

def read_rr_data(path: str) -> np.ndarray:
    """读取RR间期数据（毫秒）"""
    data = pd.read_csv(path, header=None)
    if data.shape[1] < 1:
        raise ValueError("RR文件需要至少一列数据")
    rr_ms = data.iloc[:, 0].to_numpy(dtype=float)
    
    return rr_ms

def calculate_hrv_features(rr_ms: np.ndarray) -> dict:
    """计算指定7项HRV特征。输入单位: ms - 优化版本"""
    if rr_ms is None or len(rr_ms) < 2:
        return {}

    feat = {}

    # --- 时域特征 - 向量化计算 ---
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

    # --- Poincaré特征 - 向量化计算 ---
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

    # --- SampEn --- 高效版本，使用近似算法
    sampen = np.nan
    try:
        if len(rr_ms) >= 10:
            # 使用更快的近似算法
            m = 2
            r = 0.2 * np.std(rr_ms)
            
            # 限制数据长度以提高性能
            max_len = min(len(rr_ms), 500)  # 进一步限制长度
            rr_subset = rr_ms[:max_len]
            
            # 使用滑动窗口方法
            count_m = 0
            count_m1 = 0
            total_m = 0
            total_m1 = 0
            
            # 预计算所有窗口
            windows_m = np.array([rr_subset[i:i+m] for i in range(len(rr_subset) - m)])
            windows_m1 = np.array([rr_subset[i:i+m+1] for i in range(len(rr_subset) - m - 1)])
            
            # 向量化比较
            for i in range(len(windows_m)):
                # 限制比较范围以提高性能
                end_idx = min(i + 30, len(windows_m))
                
                for j in range(i + 1, end_idx):
                    if np.max(np.abs(windows_m[i] - windows_m[j])) <= r:
                        count_m += 1
                    total_m += 1
                    
                    if i < len(windows_m1) and j < len(windows_m1):
                        if np.max(np.abs(windows_m1[i] - windows_m1[j])) <= r:
                            count_m1 += 1
                        total_m1 += 1
            
            if count_m > 0 and count_m1 > 0 and total_m > 0 and total_m1 > 0:
                sampen = -np.log((count_m1 / total_m1) / (count_m / total_m))
    except Exception:
        pass
    feat['SampEn'] = sampen

    return feat

# --------------------------------------------------------------------------------------------
# 多线程处理类
# --------------------------------------------------------------------------------------------

class MultiThreadHRVProcessor:
    def __init__(self, input_dir: str, output_file: str, max_workers: int = 8):
        self.input_dir = input_dir
        self.output_file = output_file
        self.max_workers = max_workers
        self.lock = threading.Lock()  # 用于保护文件写入操作
        self.results = []  # 存储所有处理结果
        self.processing_status = {}  # 存储每个文件的处理状态
        self.thread_status = {}  # 存储每个线程的状态
        self.monitor_running = False  # 监控线程运行状态
        self.performance_stats = {
            'start_time': 0,
            'processed_files': 0,
            'total_files': 0,
            'cpu_usage': 0,
            'memory_usage': 0
        }
        self.rr_backup_enabled = bool(RR_BACKUP_BASE_DIR.strip())
        self.rr_backup_dirs = {}  # 存储每个标签的备份目录
        
        # 确保输出目录存在
        output_dir = os.path.dirname(output_file) if os.path.dirname(output_file) else "."
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
    
    def process_single_csv(self, csv_file: str, emotion_label: str, thread_id: int) -> pd.DataFrame:
        """处理单个CSV文件，先进行智能数据段处理，然后计算HRV特征"""
        filename = os.path.basename(csv_file)
        
        try:
            # 批量更新状态，减少锁操作
            self._update_status_batch(thread_id, filename, "读取中")
            
            # 读取数据
            data = pd.read_csv(csv_file, header=None)
            
            if data.shape[1] < 2:
                self._update_status_batch(thread_id, filename, "格式错误")
                print(f"\n❌ 文件格式错误: {filename} (需要至少2列数据)")
                return pd.DataFrame()
            
            # 设置列名
            data = data.iloc[:, :2].copy()
            data.columns = ['时间', '数值']
            
            if len(data) < 200:  # 40秒需要更多数据点
                self._update_status_batch(thread_id, filename, "数据不足")
                print(f"\n❌ 数据不足: {filename} (需要至少200个数据点)")
                return pd.DataFrame()
            
            # 批量更新状态
            self._update_status_batch(thread_id, filename, "峰值检测中")
            
            # 第1步：峰值检测
            peak_info, _ = detect_all_peaks(data)
            if peak_info is None:
                self._update_status_batch(thread_id, filename, "峰值检测失败")
                print(f"\n❌ 峰值检测失败: {filename}")
                return pd.DataFrame()
            
            # 批量更新状态
            self._update_status_batch(thread_id, filename, "生成数据段中")
            
            # 第2步：生成候选数据段
            candidates = generate_candidate_segments(data, peak_info)
            if not candidates:
                self._update_status_batch(thread_id, filename, "无高质量数据段")
                print(f"\n❌ 无高质量数据段: {filename}")
                return pd.DataFrame()
            
            # 批量更新状态
            self._update_status_batch(thread_id, filename, "计算HRV特征中")
            
            # 第3步：对每个高质量数据段计算HRV特征
            results = []
            for i, segment in enumerate(candidates):
                # 从数据段中提取RR间期
                if len(segment['peak_times']) > 1:
                    rr_intervals = np.diff(segment['peak_times'])
                    
                    # 备份RR间隔数据
                    backup_path = self._backup_rr_data(rr_intervals, filename, emotion_label, i+1)
                    
                    # 计算HRV特征
                    feat = calculate_hrv_features(rr_intervals)
                    
                    if feat:
                        # 构建结果行
                        row = {
                            'file': f"{filename}_seg{i+1}",
                            'RMSSD': feat.get('RMSSD', np.nan),
                            'pNN58': feat.get('pNN58', np.nan),
                            'SDNN': feat.get('SDNN', np.nan),
                            'SD1': feat.get('SD1', np.nan),
                            'SD2': feat.get('SD2', np.nan),
                            'SD1_SD2': feat.get('SD1_SD2', np.nan),
                            'SampEn': feat.get('SampEn', np.nan),
                            'emotion': emotion_label
                        }
                        results.append(row)
            
            if not results:
                self._update_status_batch(thread_id, filename, "HRV计算失败")
                print(f"\n❌ HRV特征计算失败: {filename}")
                return pd.DataFrame()
            
            # 批量更新状态
            self._update_status_batch(thread_id, filename, f"已完成({len(results)}段)")
            
            return pd.DataFrame(results)
            
        except Exception as e:
            self._update_status_batch(thread_id, filename, "处理失败")
            print(f"\n❌ 处理失败: {filename} -> {e}")
            return pd.DataFrame()
    
    def _update_status_batch(self, thread_id: int, filename: str, status: str):
        """批量更新状态，减少锁操作频率"""
        with self.lock:
            self.thread_status[thread_id] = f"{status} {filename}"
            self.processing_status[filename] = status
    
    def _get_performance_stats(self):
        """获取性能统计信息"""
        try:
            import psutil
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory_percent = psutil.virtual_memory().percent
            return cpu_percent, memory_percent
        except:
            return 0, 0
    
    def _update_performance_stats(self, processed_count: int):
        """更新性能统计"""
        with self.lock:
            self.performance_stats['processed_files'] = processed_count
            cpu_usage, memory_usage = self._get_performance_stats()
            self.performance_stats['cpu_usage'] = cpu_usage
            self.performance_stats['memory_usage'] = memory_usage
    
    def _setup_rr_backup_dir(self, emotion_label: str) -> str:
        """为每个标签设置RR数据备份目录"""
        if not self.rr_backup_enabled:
            return None
        
        # 清理标签名称，移除特殊字符
        clean_label = emotion_label.replace(" ", "_").replace("/", "_").replace("\\", "_")
        backup_dir = os.path.join(RR_BACKUP_BASE_DIR, clean_label)
        
        # 确保目录存在（使用exist_ok=True避免目录已存在的错误）
        try:
            os.makedirs(backup_dir, exist_ok=True)
        except Exception as e:
            print(f"⚠️ 创建备份目录失败: {backup_dir}, 错误: {e}")
            return None
        
        return backup_dir
    
    def _backup_rr_data(self, rr_intervals: np.ndarray, filename: str, emotion_label: str, segment_index: int) -> str:
        """备份RR间隔数据到指定目录"""
        if not self.rr_backup_enabled:
            return None
        
        # 获取或创建备份目录
        if emotion_label not in self.rr_backup_dirs:
            self.rr_backup_dirs[emotion_label] = self._setup_rr_backup_dir(emotion_label)
        
        backup_dir = self.rr_backup_dirs[emotion_label]
        if not backup_dir:
            return None
        
        # 生成备份文件名
        base_name = os.path.splitext(filename)[0]
        backup_filename = f"{base_name}_seg{segment_index}_RR_40s.csv"
        backup_path = os.path.join(backup_dir, backup_filename)
        
        try:
            # 保存RR间隔数据
            pd.DataFrame(rr_intervals).to_csv(backup_path, index=False, header=False)
            return backup_path
        except Exception as e:
            print(f"\n⚠️ RR数据备份失败: {backup_path} - {e}")
            return None
    
    def process_all_files(self, csv_files: List[str], emotion_labels: List[str]) -> None:
        """多线程处理所有CSV文件"""
        if len(csv_files) != len(emotion_labels):
            print(f"❌ 错误: CSV文件数量({len(csv_files)})与标签数量({len(emotion_labels)})不匹配")
            return
        
        print(f"📊 开始多线程处理 {len(csv_files)} 个文件 (40秒数据段)")
        print(f"🔧 使用 {self.max_workers} 个线程")
        print()
        
        # 默认启用实时监控
        monitor_thread = self._start_status_monitor()
        print("✅ 实时监控已启动，按 Ctrl+C 可停止监控")
        time.sleep(1)
        
        start_time = time.time()
        success_count = 0
        
        # 初始化性能统计
        self.performance_stats['start_time'] = start_time
        self.performance_stats['total_files'] = len(csv_files)
        
        # 初始化处理状态
        for csv_file in csv_files:
            self.processing_status[os.path.basename(csv_file)] = "等待中"
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # 提交所有任务，为每个任务分配线程ID
            future_to_file = {}
            for i, (csv_file, emotion_label) in enumerate(zip(csv_files, emotion_labels)):
                future = executor.submit(self.process_single_csv, csv_file, emotion_label, i % self.max_workers)
                future_to_file[future] = (csv_file, emotion_label)
            
            # 处理完成的任务
            completed_count = 0
            for future in as_completed(future_to_file):
                csv_file, emotion_label = future_to_file[future]
                filename = os.path.basename(csv_file)
                
                try:
                    result = future.result()
                    if not result.empty:
                        with self.lock:
                            self.results.append(result)
                        success_count += 1
                        self.processing_status[filename] = "✅ 成功"
                    else:
                        self.processing_status[filename] = "❌ 失败"
                except Exception as e:
                    print(f"\n❌ 任务执行异常: {filename} - {e}")
                    self.processing_status[filename] = "❌ 异常"
                
                completed_count += 1
                
                # 显示进度
                if not self.monitor_running:  # 如果没有启用实时监控，显示简单进度
                    progress = (completed_count / len(csv_files)) * 100
                    # 更新性能统计
                    self._update_performance_stats(completed_count)
                    
                    # 计算处理速度
                    elapsed_time = time.time() - start_time
                    files_per_second = completed_count / elapsed_time if elapsed_time > 0 else 0
                    
                    print(f"\r🔄 处理进度: {completed_count}/{len(csv_files)} ({progress:.1f}%) - 成功: {success_count}, 失败: {completed_count - success_count} - 速度: {files_per_second:.1f} 文件/秒 - CPU: {self.performance_stats['cpu_usage']:.1f}%", end="", flush=True)
        
        end_time = time.time()
        
        # 停止监控
        if monitor_thread:
            self._stop_status_monitor()
            time.sleep(1)  # 等待监控线程结束
        
        print(f"\n=== 处理完成 ===")
        print(f"总文件数: {len(csv_files)}")
        print(f"成功处理: {success_count}")
        print(f"失败文件: {len(csv_files) - success_count}")
        print(f"处理时间: {end_time - start_time:.2f} 秒")
        
        # 显示详细处理结果
        self._show_detailed_results()
    
    def _show_detailed_results(self):
        """显示详细处理结果"""
        print(f"\n📋 详细处理结果:")
        print("-" * 80)
        
        # 按状态分组显示
        status_groups = {}
        for filename, status in self.processing_status.items():
            if status not in status_groups:
                status_groups[status] = []
            status_groups[status].append(filename)
        
        for status, files in status_groups.items():
            print(f"{status}: {len(files)} 个文件")
            for filename in sorted(files):
                print(f"  - {filename}")
            print()
    
    def _start_status_monitor(self):
        """启动状态监控线程"""
        def monitor():
            while self.monitor_running:
                with self.lock:
                    # 清屏并显示当前状态
                    os.system('cls' if os.name == 'nt' else 'clear')
                    print("🔄 实时处理状态监控 (40秒数据段)")
                    print("=" * 60)
                    
                    # 显示线程状态
                    print("🧵 线程状态:")
                    for thread_id in range(self.max_workers):
                        status = self.thread_status.get(thread_id, "空闲")
                        print(f"  线程 {thread_id}: {status}")
                    
                    print("\n📁 文件处理状态:")
                    # 按状态分组显示
                    status_groups = {}
                    for filename, status in self.processing_status.items():
                        if status not in status_groups:
                            status_groups[status] = []
                        status_groups[status].append(filename)
                    
                    for status, files in status_groups.items():
                        print(f"  {status}: {len(files)} 个文件")
                        for filename in sorted(files)[:5]:  # 只显示前5个
                            print(f"    - {filename}")
                        if len(files) > 5:
                            print(f"    ... 还有 {len(files) - 5} 个文件")
                    
                    print(f"\n⏱️  按 Ctrl+C 停止监控")
                
                time.sleep(2)  # 每2秒更新一次
        
        self.monitor_running = True
        monitor_thread = threading.Thread(target=monitor, daemon=True)
        monitor_thread.start()
        return monitor_thread
    
    def _stop_status_monitor(self):
        """停止状态监控"""
        self.monitor_running = False
    
    def save_results(self) -> None:
        """保存结果到CSV文件"""
        if not self.results:
            print("❌ 没有结果需要保存")
            return
        
        # 合并所有结果
        df = pd.concat(self.results, ignore_index=True)
        
        # 按指定顺序排列列
        cols = ['file', 'RMSSD', 'pNN58', 'SDNN', 'SD1', 'SD2', 'SD1_SD2', 'SampEn', 'emotion']
        for c in cols:
            if c not in df.columns:
                df[c] = np.nan
        
        # 重新排列列顺序
        df = df[cols]
        
        # 检查文件是否已存在，决定是追加还是新建
        file_exists = os.path.exists(self.output_file)
        
        if file_exists:
            # 追加模式：不写表头，只追加数据行
            df.to_csv(self.output_file, mode='a', header=False, index=False, encoding='utf-8-sig')
            print(f"📝 新记录已追加到现有文件: {self.output_file}")
        else:
            # 新建模式：写表头和数据
            df.to_csv(self.output_file, index=False, encoding='utf-8-sig')
            print(f"📝 HRV特征数据集已创建: {self.output_file}")
        
        print(f"📊 数据集形状: {df.shape}")
        print(f"🔢 特征数量: {len(df.columns) - 2}")  # 减去file和emotion列
        print(f"📈 样本数量: {len(df)}")
        
        # 显示特征列信息
        print(f"🏷️ 特征列: {list(df.columns[1:-1])}")  # 除了file和emotion
        print(f"🏷️ 标签列: {df.columns[-1]}")
        
        # 显示前几行数据
        print(f"\n📋 数据集前3行:")
        print(df.head(3))
        
        # 显示RR数据备份信息
        if self.rr_backup_enabled and self.rr_backup_dirs:
            print(f"\n💾 RR间隔数据备份信息 (40秒版本):")
            for emotion_label, backup_dir in self.rr_backup_dirs.items():
                if backup_dir and os.path.exists(backup_dir):
                    backup_files = glob.glob(os.path.join(backup_dir, "*_RR_40s.csv"))
                    print(f"   {emotion_label}: {len(backup_files)} 个文件 -> {backup_dir}")
        elif self.rr_backup_enabled:
            print(f"\n⚠️ RR数据备份功能已启用，但未设置备份目录路径")
        else:
            print(f"\n💾 RR数据备份功能未启用")

# --------------------------------------------------------------------------------------------
# 标签输入处理
# --------------------------------------------------------------------------------------------

def get_emotion_labels() -> List[str]:
    """从用户输入获取情绪标签列表"""
    print("请输入标签列表（第一个是姓名，后续是情绪标签）:")
    print("格式示例:")
    print("张三")
    print("愉悦")
    print("平静")
    print("焦虑")
    print("悲伤")
    print("...")
    print("输入完成后按Ctrl+Z然后回车结束")
    print()
    
    labels = []
    try:
        while True:
            line = input().strip()
            if line:  # 忽略空行
                labels.append(line)
    except EOFError:
        pass
    
    return labels

# --------------------------------------------------------------------------------------------
# 主处理函数
# --------------------------------------------------------------------------------------------

def main():
    print("=== 多线程HRV特征计算脚本 - 40秒版本 ===")
    print(f"输入数据目录: {INPUT_DATA_DIR}")
    print(f"输出文件: {OUTPUT_FILE}")
    print(f"最大线程数: {MAX_WORKERS}")
    print(f"数据段长度: {SEGMENT_DURATION_MS/1000:.1f} 秒")
    
    # 显示RR备份功能状态
    if RR_BACKUP_BASE_DIR.strip():
        print(f"RR数据备份: 已启用 -> {RR_BACKUP_BASE_DIR}")
    else:
        print(f"RR数据备份: 未启用 (请在脚本中设置 RR_BACKUP_BASE_DIR)")
    print()
    
    # 检查输入目录
    if not os.path.exists(INPUT_DATA_DIR):
        print(f"❌ 输入目录不存在: {INPUT_DATA_DIR}")
        return
    
    # 查找所有CSV文件
    csv_files = sorted(glob.glob(os.path.join(INPUT_DATA_DIR, "*.csv")))
    if not csv_files:
        print(f"❌ 在 {INPUT_DATA_DIR} 中未找到任何CSV文件")
        return
    
    print(f"📁 找到 {len(csv_files)} 个CSV文件:")
    for i, csv_file in enumerate(csv_files, 1):
        print(f"   {i}. {os.path.basename(csv_file)}")
    print()
    
    # 获取标签列表
    labels = get_emotion_labels()
    
    if not labels:
        print("❌ 未输入任何标签")
        return
    
    if len(labels) < 2:
        print("❌ 至少需要输入姓名和一个情绪标签")
        return
    
    # 检查标签数量是否匹配
    if len(labels) - 1 != len(csv_files):
        print(f"❌ 标签数量不匹配:")
        print(f"   CSV文件数量: {len(csv_files)}")
        print(f"   情绪标签数量: {len(labels) - 1}")
        print(f"   请确保情绪标签数量与CSV文件数量一致")
        return
    
    # 构建完整的情绪标签（姓名 + 情绪）
    name = labels[0]
    emotion_labels = [f"{name} {emotion}" for emotion in labels[1:]]
    
    print(f"👤 姓名: {name}")
    print(f"🏷️ 情绪标签:")
    for i, emotion_label in enumerate(emotion_labels, 1):
        print(f"   {i}. {emotion_label}")
    print()
    
    # 创建处理器并开始处理
    processor = MultiThreadHRVProcessor(INPUT_DATA_DIR, OUTPUT_FILE, MAX_WORKERS)
    processor.process_all_files(csv_files, emotion_labels)
    
    # 保存结果
    processor.save_results()
    
    print(f"\n🎉 所有处理完成！")

# --------------------------------------------------------------------------------------------
# 使用示例和说明
# --------------------------------------------------------------------------------------------

def show_usage_example():
    """显示使用示例"""
    print("=== 使用示例 - 40秒版本 ===")
    print("1. 确保 processed_data 目录下有已划分好的CSV文件")
    print("2. 运行脚本后按提示输入标签:")
    print("   张三")
    print("   愉悦")
    print("   平静")
    print("   焦虑")
    print("   悲伤")
    print("3. 脚本会自动:")
    print("   - 对每个CSV文件进行峰值检测")
    print("   - 使用滑动窗口生成40s高质量数据段")
    print("   - 对每个数据段计算HRV特征")
    print("   - 多线程并行处理所有文件")
    print("   - 将结果追加到 hrv_data_40s.csv")
    print("   - 备份RR间隔数据到40sdata/backup目录（如果启用）")
    print("4. 输出格式: 张三 愉悦, 张三 平静, 张三 焦虑, 张三 悲伤")
    print("5. 每个CSV文件可能产生多个数据段，文件名格式: 原文件名_seg1, 原文件名_seg2...")
    print("6. RR数据备份: 每个标签的数据单独保存到不同目录，文件名包含_40s标识")

if __name__ == "__main__":
    # 显示使用说明
    show_usage_example()
    print("\n" + "="*60 + "\n")
    
    # 运行主程序
    main()
    subprocess.run(["clear_processed_data.bat"])