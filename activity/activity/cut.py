import pandas as pd
import numpy as np
import sys
import os

def slice_time_series(df, feature_cols, window_length=200, stride=100):
    """对时序数据进行滑动窗口切片"""
    slices = []
    
    # 按activity、subject分组处理（每个组合是一条完整时序数据）
    grouped = df.groupby(['activity', 'subject'])
    
    for (activity, subject), group in grouped:
        label = group['label'].iloc[0]
        feature_data = group[feature_cols].values
        
        if len(feature_data) < window_length:
            continue
        
        for i in range(0, len(feature_data) - window_length + 1, stride):
            window = feature_data[i:i+window_length]
            slices.append({
                'label': label,
                'activity': activity,
                'subject': subject,
                'window_idx': i,
                'data': window
            })
    
    return slices

def save_window_csv(window_data, label, activity, subject, window_idx, output_dir):
    """保存单个窗口为CSV文件"""
    df_window = pd.DataFrame(
        window_data,
        columns=['userAcceleration.x', 'userAcceleration.y', 'userAcceleration.z',
                 'rotationRate.x', 'rotationRate.y', 'rotationRate.z']
    )
    
    df_window['label'] = label
    df_window['subject'] = subject
    
    filename = f"{activity}_{subject}_{window_idx}.csv"
    filepath = os.path.join(output_dir, str(label), filename)
    df_window.to_csv(filepath, index=False)
    
    return filepath

def main():
    input_file = 'cleaned_data.csv'
    output_dir = 'ori_data'
    
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
    if len(sys.argv) > 2:
        output_dir = sys.argv[2]
    
    print(f"读取数据: {input_file}")
    df = pd.read_csv(input_file)
    print(f"数据量: {len(df)}")
    
    feature_cols = [
        'userAcceleration.x',
        'userAcceleration.y',
        'userAcceleration.z',
        'rotationRate.x',
        'rotationRate.y',
        'rotationRate.z'
    ]
    
    # 创建输出目录
    os.makedirs(os.path.join(output_dir, '0'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, '1'), exist_ok=True)
    print(f"\n创建输出目录: {output_dir}/0 和 {output_dir}/1")
    
    # 时序切片
    print("\n进行时序数据切片...")
    print(f"窗口长度: 200, 滑动长度: 100")
    slices = slice_time_series(df, feature_cols, window_length=200, stride=100)
    print(f"生成切片数: {len(slices)}")
    
    # 保存每个窗口为独立的CSV文件
    print("\n保存窗口文件...")
    label_counts = {0: 0, 1: 0}
    
    for idx, slice_info in enumerate(slices):
        filepath = save_window_csv(
            slice_info['data'],
            slice_info['label'],
            slice_info['activity'],
            slice_info['subject'],
            slice_info['window_idx'],
            output_dir
        )
        label_counts[slice_info['label']] += 1
        
        if (idx + 1) % 500 == 0:
            print(f"已保存 {idx + 1}/{len(slices)} 个窗口...")
    
    print(f"\n所有窗口已保存到: {output_dir}/")
    
    print("\n数据统计:")
    print(f"总切片数: {len(slices)}")
    print(f"每个切片: 200行 × 6维 + 标签列 + 受试者编号列")
    print(f"正类切片数（保存到 {output_dir}/1/）: {label_counts[1]}")
    print(f"负类切片数（保存到 {output_dir}/0/）: {label_counts[0]}")

if __name__ == '__main__':
    main()

