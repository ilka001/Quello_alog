#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试HRV特征提取器的输出格式 - 简化版本
"""

def test_output_format():
    """测试输出格式"""
    print("测试HRV特征提取器输出格式")
    print("=" * 40)
    
    # 模拟特征提取结果
    features = {
        'RMSSD': 45.2341,
        'pNN58': 12.8456,
        'SDNN': 52.3789,
        'SD1': 23.1567,
        'SD2': 29.2341,
        'SD1_SD2': 0.7921
    }
    
    print("提取的6个HRV特征:")
    feature_names = ['RMSSD', 'pNN58', 'SDNN', 'SD1', 'SD2', 'SD1_SD2']
    for name in feature_names:
        print(f"   {name}: {features[name]:.4f}")
    
    # 生成输出行（按顺序）
    output_values = [features[name] for name in feature_names]
    output_line = ','.join([f"{value:.4f}" for value in output_values])
    
    print(f"\n输出行格式:")
    print(f"'{output_line}'")
    
    # 保存到文件
    output_file = 'test_output.csv'
    with open(output_file, 'w', encoding='utf-8-sig') as f:
        f.write(output_line)
    
    print(f"\n保存到文件: {output_file}")
    print("文件内容:")
    with open(output_file, 'r', encoding='utf-8-sig') as f:
        content = f.read().strip()
        print(f"'{content}'")
    
    print(f"\n✅ 输出格式测试完成！")
    print(f"📊 输出格式: 一行6个特征值，逗号分隔，无表头")
    print(f"📋 特征顺序: RMSSD, pNN58, SDNN, SD1, SD2, SD1_SD2")

if __name__ == "__main__":
    test_output_format()

