import pandas as pd
import numpy as np
import os
from scipy import signal

def remove_outliers(df, feature_cols, n_std=3):
    """删除超出n倍标准差的异常值"""
    df_clean = df.copy()
    
    for col in feature_cols:
        if col in df_clean.columns:
            mean_val = df_clean[col].mean()
            std_val = df_clean[col].std()
            lower_bound = mean_val - n_std * std_val
            upper_bound = mean_val + n_std * std_val
            
            outliers = (df_clean[col] < lower_bound) | (df_clean[col] > upper_bound)
            df_clean = df_clean[~outliers]
    
    return df_clean.reset_index(drop=True)

def denoise_data(df, feature_cols, window_size=5):
    """使用滑动窗口均值滤波进行降噪"""
    df_denoised = df.copy()
    for col in feature_cols:
        if col in df_denoised.columns:
            df_denoised[col] = df_denoised[col].rolling(window=window_size, center=True, min_periods=1).mean()
    return df_denoised

def slice_time_series(df, feature_cols, window_length=200, stride=100):
    """对时序数据进行滑动窗口切片"""
    slices = []
    
    # 按activity、subject分组处理（每个组合是一条完整时序数据）
    grouped = df.groupby(['activity', 'subject'])
    
    for (activity, subject), group in grouped:
        label = group['label'].iloc[0]  # 同一组的label相同
        feature_data = group[feature_cols].values
        
        # 如果数据长度不足窗口长度，跳过
        if len(feature_data) < window_length:
            continue
        
        # 滑动窗口切片
        for i in range(0, len(feature_data) - window_length + 1, stride):
            window = feature_data[i:i+window_length]  # shape: (200, 6)
            slices.append({
                'label': label,
                'activity': activity,
                'subject': subject,
                'window_idx': i,
                'data': window.flatten()  # 展平为 1200 维 (6*200)
            })
    
    return slices

def process_activity_data(base_path, activity_dirs, label):
    """处理活动数据"""
    all_data = []
    
    for activity_dir in activity_dirs:
        activity_path = os.path.join(base_path, activity_dir)
        csv_files = [f for f in os.listdir(activity_path) if f.endswith('.csv')]
        
        for csv_file in sorted(csv_files):
            file_path = os.path.join(activity_path, csv_file)
            df = pd.read_csv(file_path)
            
            # 提取6维特征
            features = [
                'userAcceleration.x',
                'userAcceleration.y', 
                'userAcceleration.z',
                'rotationRate.x',
                'rotationRate.y',
                'rotationRate.z'
            ]
            
            # 检查列是否存在
            available_features = [f for f in features if f in df.columns]
            if len(available_features) != 6:
                print(f"警告: {file_path} 缺少某些特征列")
                continue
            
            # 提取特征列
            feature_df = df[available_features].copy()
            
            # 添加标签
            feature_df['label'] = label
            feature_df['activity'] = activity_dir
            feature_df['subject'] = csv_file.replace('.csv', '')
            
            all_data.append(feature_df)
    
    return pd.concat(all_data, ignore_index=True) if all_data else pd.DataFrame()

def main():
    base_path = 'data/A_DeviceMotion_data/A_DeviceMotion_data'
    
    # 定义活动类型和标签
    jog_dirs = ['jog_9', 'jog_16']
    sit_dirs = ['sit_5', 'sit_13']
    std_dirs = ['std_6', 'std_14']
    
    print("开始处理数据...")
    
    # 处理正类（跑步）
    print("处理正类数据（jog）...")
    jog_data = process_activity_data(base_path, jog_dirs, label=1)
    print(f"正类数据量: {len(jog_data)}")
    
    # 处理负类（静止：sit和std）
    print("处理负类数据（sit）...")
    sit_data = process_activity_data(base_path, sit_dirs, label=0)
    print(f"sit数据量: {len(sit_data)}")
    
    print("处理负类数据（std）...")
    std_data = process_activity_data(base_path, std_dirs, label=0)
    print(f"std数据量: {len(std_data)}")
    
    # 合并数据
    all_data = pd.concat([jog_data, sit_data, std_data], ignore_index=True)
    print(f"总数据量: {len(all_data)}")
    print(f"正类数量: {(all_data['label']==1).sum()}")
    print(f"负类数量: {(all_data['label']==0).sum()}")
    
    # 提取特征列（不包括metadata列）
    feature_cols = [
        'userAcceleration.x',
        'userAcceleration.y',
        'userAcceleration.z',
        'rotationRate.x',
        'rotationRate.y',
        'rotationRate.z'
    ]
    
    # 保存筛选并编码的数据（用于clean.py输入）
    processed_data = all_data.copy()
    output_path = 'processed_data.csv'
    processed_data.to_csv(output_path, index=False)
    print(f"筛选并编码的数据已保存到: {output_path}")
    
    # 异常值处理
    print("\n进行异常值处理（删除超出3倍标准差的数据）...")
    before_outlier = len(processed_data)
    processed_data = remove_outliers(processed_data, feature_cols, n_std=3)
    after_outlier = len(processed_data)
    print(f"异常值删除: {before_outlier} -> {after_outlier} (删除了 {before_outlier - after_outlier} 条)")
    
    # 降噪处理
    print("进行降噪处理...")
    processed_data = denoise_data(processed_data, feature_cols, window_size=5)
    
    # 时序切片
    print("\n进行时序数据切片...")
    print(f"窗口长度: 200, 滑动长度: 100")
    slices = slice_time_series(processed_data, feature_cols, window_length=200, stride=100)
    print(f"生成切片数: {len(slices)}")
    
    # 构建切片数据框
    slice_data = []
    for slice_info in slices:
        row = {
            'label': slice_info['label'],
            'activity': slice_info['activity'],
            'subject': slice_info['subject'],
            'window_idx': slice_info['window_idx']
        }
        # 添加1200个特征列（6维*200点）
        for i, val in enumerate(slice_info['data']):
            row[f'f_{i}'] = val
        slice_data.append(row)
    
    slice_df = pd.DataFrame(slice_data)
    
    # 保存切片数据
    ori_data_path = 'ori_data.csv'
    slice_df.to_csv(ori_data_path, index=False)
    print(f"\n切片数据已保存到: {ori_data_path}")
    
    # 保存统计信息
    print("\n数据统计:")
    print(f"总切片数: {len(slice_df)}")
    print(f"每个切片维度: 6维*200点 = 1200维")
    print(f"正类切片数: {(slice_df['label']==1).sum()}")
    print(f"负类切片数: {(slice_df['label']==0).sum()}")

if __name__ == '__main__':
    main()

