import pandas as pd
import numpy as np
import os
from scipy import signal
from scipy.fft import fft, fftfreq

def extract_time_domain_features(data):
    """提取时域特征（12维）"""
    features = {}
    
    # 提取6维数据
    acc_x = data['userAcceleration.x'].values
    acc_y = data['userAcceleration.y'].values
    acc_z = data['userAcceleration.z'].values
    rot_x = data['rotationRate.x'].values
    rot_y = data['rotationRate.y'].values
    rot_z = data['rotationRate.z'].values
    
    # 单轴统计特征（6维）- 每个轴计算标准差（最核心的区分特征）
    features['acc_x_std'] = np.std(acc_x)
    features['acc_y_std'] = np.std(acc_y)
    features['acc_z_std'] = np.std(acc_z)
    features['rot_x_std'] = np.std(rot_x)
    features['rot_y_std'] = np.std(rot_y)
    features['rot_z_std'] = np.std(rot_z)
    
    # 计算加速度模长
    acc_magnitude = np.sqrt(acc_x**2 + acc_y**2 + acc_z**2)
    features['acc_mag_mean'] = np.mean(acc_magnitude)
    features['acc_mag_std'] = np.std(acc_magnitude)
    features['acc_mag_peak'] = np.max(acc_magnitude) - np.min(acc_magnitude)
    
    # 计算旋转速率模长
    rot_magnitude = np.sqrt(rot_x**2 + rot_y**2 + rot_z**2)
    features['rot_mag_mean'] = np.mean(rot_magnitude)
    features['rot_mag_std'] = np.std(rot_magnitude)
    features['rot_mag_peak'] = np.max(rot_magnitude) - np.min(rot_magnitude)
    
    return features

def extract_frequency_domain_features(data):
    """提取频域特征（4维）"""
    features = {}
    
    # 提取数据
    acc_x = data['userAcceleration.x'].values
    acc_y = data['userAcceleration.y'].values
    acc_z = data['userAcceleration.z'].values
    rot_x = data['rotationRate.x'].values
    rot_y = data['rotationRate.y'].values
    rot_z = data['rotationRate.z'].values
    
    # 计算模长
    acc_magnitude = np.sqrt(acc_x**2 + acc_y**2 + acc_z**2)
    rot_magnitude = np.sqrt(rot_x**2 + rot_y**2 + rot_z**2)
    
    # 假设采样频率（需要根据实际情况调整，这里假设50Hz）
    # 200个采样点，如果采样频率为50Hz，则时长4秒
    fs = 50.0  # 采样频率
    N = len(acc_magnitude)
    
    # 加速度模长的FFT
    acc_fft = np.abs(fft(acc_magnitude))
    acc_freqs = fftfreq(N, 1/fs)
    acc_freqs = acc_freqs[:N//2]
    acc_fft = acc_fft[:N//2]
    
    # 找到主频（能量最高的频率，排除0Hz）
    acc_fft_positive = acc_fft[acc_freqs > 0]
    acc_freqs_positive = acc_freqs[acc_freqs > 0]
    if len(acc_fft_positive) > 0:
        main_freq_idx = np.argmax(acc_fft_positive)
        features['acc_main_freq'] = acc_freqs_positive[main_freq_idx]
        features['acc_main_freq_amp'] = acc_fft_positive[main_freq_idx]
    else:
        features['acc_main_freq'] = 0
        features['acc_main_freq_amp'] = 0
    
    # 旋转速率模长的FFT
    rot_fft = np.abs(fft(rot_magnitude))
    rot_freqs = fftfreq(N, 1/fs)
    rot_freqs = rot_freqs[:N//2]
    rot_fft = rot_fft[:N//2]
    
    # 找到主频
    rot_fft_positive = rot_fft[rot_freqs > 0]
    rot_freqs_positive = rot_freqs[rot_freqs > 0]
    if len(rot_fft_positive) > 0:
        main_freq_idx = np.argmax(rot_fft_positive)
        features['rot_main_freq'] = rot_freqs_positive[main_freq_idx]
        # 旋转速率模长的主频占比
        total_energy = np.sum(rot_fft_positive)
        features['rot_main_freq_ratio'] = rot_fft_positive[main_freq_idx] / total_energy if total_energy > 0 else 0
    else:
        features['rot_main_freq'] = 0
        features['rot_main_freq_ratio'] = 0
    
    return features

def extract_temporal_features(data):
    """提取时序形态特征（2维）"""
    features = {}
    
    # 提取数据
    acc_x = data['userAcceleration.x'].values
    acc_y = data['userAcceleration.y'].values
    acc_z = data['userAcceleration.z'].values
    rot_x = data['rotationRate.x'].values
    rot_y = data['rotationRate.y'].values
    rot_z = data['rotationRate.z'].values
    
    # 计算加速度模长
    acc_magnitude = np.sqrt(acc_x**2 + acc_y**2 + acc_z**2)
    
    # 计算旋转速率模长（用于零交叉率）
    rot_magnitude = np.sqrt(rot_x**2 + rot_y**2 + rot_z**2)
    
    # 1. 加速度模长的波峰数量
    # 找到峰值（使用阈值，阈值为均值的1.5倍）
    threshold = np.mean(acc_magnitude) * 1.5
    peaks, _ = signal.find_peaks(acc_magnitude, height=threshold)
    features['acc_peak_count'] = len(peaks)
    
    # 2. 旋转速率的零交叉率
    # 计算零交叉次数
    zero_crossings = np.where(np.diff(np.signbit(rot_magnitude - np.mean(rot_magnitude))))[0]
    features['rot_zero_crossing_rate'] = len(zero_crossings)
    
    return features

def extract_features_from_window(filepath):
    """从单个窗口CSV文件提取特征"""
    df = pd.read_csv(filepath)
    
    # 提取所有特征
    time_features = extract_time_domain_features(df)
    freq_features = extract_frequency_domain_features(df)
    temp_features = extract_temporal_features(df)
    
    # 合并特征
    all_features = {**time_features, **freq_features, **temp_features}
    
    # 获取标签和受试者编号
    all_features['label'] = df['label'].iloc[0]
    all_features['subject'] = df['subject'].iloc[0]
    
    return all_features

def main():
    input_dir = 'ori_data'
    output_file = 'fortrain.csv'
    
    print(f"从 {input_dir} 目录读取窗口文件...")
    
    all_features = []
    
    # 遍历0和1子目录
    for label in ['0', '1']:
        label_dir = os.path.join(input_dir, label)
        if not os.path.exists(label_dir):
            continue
        
        csv_files = [f for f in os.listdir(label_dir) if f.endswith('.csv')]
        print(f"\n处理标签 {label} 的文件: {len(csv_files)} 个")
        
        for idx, csv_file in enumerate(csv_files):
            filepath = os.path.join(label_dir, csv_file)
            
            features = extract_features_from_window(filepath)
            all_features.append(features)
            
            if (idx + 1) % 500 == 0:
                print(f"  已处理 {idx + 1}/{len(csv_files)} 个文件...")
    
    # 转换为DataFrame
    df_features = pd.DataFrame(all_features)
    
    # 重新排列列顺序：特征列在前，然后是label和subject
    feature_cols = [col for col in df_features.columns if col not in ['label', 'subject']]
    column_order = feature_cols + ['label', 'subject']
    df_features = df_features[column_order]
    
    # 保存
    df_features.to_csv(output_file, index=False)
    print(f"\n特征提取完成！")
    print(f"特征维度: {len(feature_cols)} 维")
    print(f"总样本数: {len(df_features)}")
    print(f"正类样本数: {(df_features['label']==1).sum()}")
    print(f"负类样本数: {(df_features['label']==0).sum()}")
    print(f"\n数据已保存到: {output_file}")
    print(f"\n特征列: {feature_cols}")

if __name__ == '__main__':
    main()

