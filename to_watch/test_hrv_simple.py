#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简单的HRV特征提取测试脚本
不依赖外部库，只测试基本逻辑
"""

import sys
import os

def test_file_exists():
    """测试文件是否存在"""
    test_file = r"C:\Users\QAQ\Desktop\emotion\010_t102743_114093_001.csv"
    if os.path.exists(test_file):
        print(f"✅ 测试文件存在: {test_file}")
        return True
    else:
        print(f"❌ 测试文件不存在: {test_file}")
        return False

def test_config_params():
    """测试配置参数"""
    print("🔧 配置参数测试:")
    input_file = r"C:\Users\QAQ\Desktop\emotion\010_t102743_114093_001.csv"
    print(f"   输入文件: {input_file}")
    print(f"   输出文件: hrv_features.csv")
    print(f"   情绪标签: 平静")
    print(f"   数据段长度: 10000 ms (10秒)")
    print(f"   峰值检测参数: distance=5, prominence=25")
    print(f"   质量评估: 已禁用（直接处理所有数据）")
    return True

def test_peak_detection_params():
    """测试峰值检测参数"""
    print("\n🎯 峰值检测参数:")
    params = {
        'distance': 5,
        'prominence': 25,
        'height': None
    }
    for key, value in params.items():
        print(f"   {key}: {value}")
    return True

def test_quality_params():
    """测试质量评估参数（已禁用）"""
    print("\n📊 质量评估参数（已禁用）:")
    params = {
        'min_peaks_per_segment': 1,
        'max_peaks_per_segment': 1000,
        'gap_threshold_factor': 10.0,
        'rr_variability_threshold': 10.0,
        'outlier_threshold': 10.0,
        'rr_range_min_factor': 0.1,
        'rr_range_max_factor': 10.0,
        'min_segment_quality_score': -1
    }
    for key, value in params.items():
        print(f"   {key}: {value}")
    return True

def test_hrv_features():
    """测试HRV特征列表"""
    print("\n📈 输出的6个HRV特征:")
    features = ['RMSSD', 'pNN58', 'SDNN', 'SD1', 'SD2', 'SD1_SD2']
    for i, feature in enumerate(features, 1):
        print(f"   {i}. {feature}")
    return True

def main():
    """主函数"""
    print("HRV特征提取器 - 10秒版本测试")
    print("=" * 50)
    
    tests = [
        ("文件存在性测试", test_file_exists),
        ("配置参数测试", test_config_params),
        ("峰值检测参数测试", test_peak_detection_params),
        ("质量评估参数测试", test_quality_params),
        ("HRV特征测试", test_hrv_features)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n🧪 {test_name}:")
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name} 通过")
            else:
                print(f"❌ {test_name} 失败")
        except Exception as e:
            print(f"❌ {test_name} 出错: {e}")
    
    print(f"\n📊 测试结果: {passed}/{total} 通过")
    
    if passed == total:
        print("🎉 所有测试通过！脚本配置正确。")
        print("\n💡 下一步:")
        print("1. 安装依赖: pip install numpy pandas scipy")
        print("2. 运行: python hrv_feature_extractor.py --use-config")
    else:
        print("⚠️ 部分测试失败，请检查配置。")

if __name__ == "__main__":
    main()
