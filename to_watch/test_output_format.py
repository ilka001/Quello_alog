#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试HRV特征提取器的输出格式
"""

import pandas as pd
import numpy as np

def test_output_format():
    """测试输出格式"""
    print("测试HRV特征提取器输出格式")
    print("=" * 40)
    
    # 模拟特征提取结果
    result = {
        'RMSSD': 45.2341,
        'pNN58': 12.8456,
        'SDNN': 52.3789,
        'SD1': 23.1567,
        'SD2': 29.2341,
        'SD1_SD2': 0.7921,
        'emotion': '平静',
        'peak_count': 8,
        'segment_duration': 10.0
    }
    
    print("原始结果字典:")
    for key, value in result.items():
        print(f"   {key}: {value}")
    
    # 创建DataFrame
    df = pd.DataFrame([result])
    
    # 按指定顺序排列列（只包含6个核心特征）
    cols = ['RMSSD', 'pNN58', 'SDNN', 'SD1', 'SD2', 'SD1_SD2']
    df = df[cols]
    
    print(f"\n提取的6个特征:")
    for col in cols:
        print(f"   {col}: {df[col].iloc[0]:.4f}")
    
    # 保存到文件（不包含表头和索引）
    output_file = 'test_output.csv'
    df.to_csv(output_file, index=False, header=False, encoding='utf-8-sig')
    
    print(f"\n保存到文件: {output_file}")
    print("文件内容:")
    with open(output_file, 'r', encoding='utf-8-sig') as f:
        content = f.read().strip()
        print(f"'{content}'")
    
    print(f"\n✅ 输出格式测试完成！")
    print(f"📊 输出格式: 一行6个特征值，逗号分隔，无表头")

if __name__ == "__main__":
    test_output_format()

