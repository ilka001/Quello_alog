import pandas as pd
import numpy as np


def convert_csv_float_to_int8_pandas(input_file, output_file):
    """
    使用pandas将CSV文件中的浮点数转换为int8
    """
    # 读取CSV文件
    df = pd.read_csv(input_file)
    
    # 遍历所有列，将浮点数列转换为int8
    for column in df.columns[:5]:
        if df[column].dtype in ['float64', 'float32']:
            # 将浮点数转换为int8
            df[column] = df[column].astype(np.int8)
    
    # 保存转换后的CSV文件
    df.to_csv(output_file, index=False)
    print(f"转换完成！结果已保存到: {output_file}")

# 使用示例


convert_csv_float_to_int8_pandas(r'C:\Users\QAQ\Desktop\emotion\hrv_MK.csv', r'C:\Users\QAQ\Desktop\emotion\hrv_MK_int8.csv')

i=1

if i == 0:
    df = pd.read_csv(r'C:\Users\QAQ\Desktop\emotion\hrv_data.csv')

    # # --- 增加新列 ---
    # # 基于已有列计算新列
    # df['Birth_Year'] = 2024 - df['Age']
    # # 或者直接添加一个固定值的列
    # df['Department'] = 'HR'

    # --- 删除列 ---
    # 方式1： 使用drop方法，axis=1表示列
    df = df.drop('SampEn', axis=1)
    df = df.drop('file', axis=1)
    # # 方式2： 直接选择要保留的列
    # df = df[['Name', 'Age']]

    # # --- 重命名列 ---
    # df = df.rename(columns={'Name': 'Full_Name', 'Age': 'Age_Years'})
    column_name = 'emotion'
    # # --- 修改特定列的数据类型 ---
    # df['Age'] = df['Age'].astype(float) # 改为浮点数
    df[column_name] = df[column_name].astype(str).str.lstrip('0 ')
    df.to_csv(r'hrv_MK.csv', index=False)




    # df = pd.read_csv('data.csv')

    # # --- 修改满足条件的行 ---
    # # 将所有在 ‘Beijing’ 的人的年龄设置为 40
    # df.loc[df['City'] == 'Beijing', 'Age'] = 40

    # # --- 复杂的条件组合 ---
    # # 将年龄大于28 并且 城市是 ‘Shanghai’ 的人的部门改为 ‘Finance’
    # df.loc[(df['Age'] > 28) & (df['City'] == 'Shanghai'), 'Department'] = 'Finance'

    # # 使用 | 表示 ‘或’ 条件
    # df.loc[(df['Age'] < 26) | (df['City'] == 'Guangzhou'), 'Bonus'] = 5000

    # df.to_csv('data_conditional_modified.csv', index=False)