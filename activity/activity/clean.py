import pandas as pd
import numpy as np
import sys

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

def main():
    input_file = 'processed_data.csv'
    output_file = 'cleaned_data.csv'
    
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
    if len(sys.argv) > 2:
        output_file = sys.argv[2]
    
    print(f"读取数据: {input_file}")
    df = pd.read_csv(input_file)
    print(f"原始数据量: {len(df)}")
    
    feature_cols = [
        'userAcceleration.x',
        'userAcceleration.y',
        'userAcceleration.z',
        'rotationRate.x',
        'rotationRate.y',
        'rotationRate.z'
    ]
    
    # 异常值处理
    print("\n进行异常值处理（删除超出3倍标准差的数据）...")
    before_outlier = len(df)
    df = remove_outliers(df, feature_cols, n_std=3)
    after_outlier = len(df)
    print(f"异常值删除: {before_outlier} -> {after_outlier} (删除了 {before_outlier - after_outlier} 条)")
    
    # 降噪处理
    print("进行降噪处理...")
    df = denoise_data(df, feature_cols, window_size=5)
    
    # 保存清理后的数据
    df.to_csv(output_file, index=False)
    print(f"\n清理后的数据已保存到: {output_file}")
    
    print("\n数据统计:")
    print(f"清理后数据量: {len(df)}")
    print(f"正类数量: {(df['label']==1).sum()}")
    print(f"负类数量: {(df['label']==0).sum()}")

if __name__ == '__main__':
    main()

