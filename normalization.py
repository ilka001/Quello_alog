# 做以下两件事：
# 1. 基线计算：按被试分组，计算每个被试在平静状态下的均值和方差
# 2. Z-score归一化：把每个被试HRV数据中原始绝对的特征值，转化为相对的情绪偏移值




import pandas as pd



# --- 配置信息 ---
INPUT_FILE = 'seleted11.csv'
OUTPUT_FILE = 'seleted11_normalized.csv'



def normalize_hrv_data(input_path, output_path):
    # 1. 加载数据
    df = pd.read_csv(input_path)

    # 2. 筛选出情绪为“平静”的数据，并按照被试分组计算每个特征的基线均值和标准差

    calm_data = df[df['emotion'] == '平静']

    # 获取数值型特征列
    feature_columns = calm_data.select_dtypes(include='number').columns.tolist()

    baseline_stats = calm_data.groupby('subject')[feature_columns].agg(['mean', 'std'])
    # 3. Z-score 归一化
    df_normalized = df.copy()

    # 遍历每个被试
    for subject in df['subject'].unique():
        # 检查该被试是否存在于基线统计数据中
        if subject in baseline_stats.index:
            # 获取当前被试的均值和标准差
            subject_means = baseline_stats.loc[subject].loc[(feature_columns, 'mean')]
            subject_stds = baseline_stats.loc[subject].loc[(feature_columns, 'std')]
            
            # 提取索引，以便正确赋值
            subject_means.index = subject_means.index.get_level_values(0)
            subject_stds.index = subject_stds.index.get_level_values(0)

            # 定位当前被试在原始数据中的所有行
            subject_indices = df_normalized['subject'] == subject
            
            # 对每个特征列进行z-score归一化
            # z = (x - mean) / std
            df_normalized.loc[subject_indices, feature_columns] = \
                (df.loc[subject_indices, feature_columns] - subject_means) / subject_stds

    df_normalized.to_csv(OUTPUT_FILE, index=False)

    print(f"Z-score 归一化完成，结果保存在 '{OUTPUT_FILE}' 文件中。")



if __name__ == "__main__":
    normalize_hrv_data(INPUT_FILE, OUTPUT_FILE)
