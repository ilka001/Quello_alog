#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os

# 添加当前目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from svm_pairwise_classifier import SVMPairwiseClassifier

def main():
    """自动运行SVM训练和ONNX导出"""
    print("自动SVM训练和ONNX导出")
    print("="*50)
    
    # 初始化SVM分类器
    data_file = r'C:\Users\QAQ\Desktop\emotion\hrv_FB.csv'
    classifier = SVMPairwiseClassifier(data_file)
    
    # 加载数据
    classifier.load_data()
    
    # 自动训练所有SVM组合
    print("\n开始训练所有SVM两两组合...")
    classifier.train_all_svm_pairs()
    
    # 显示结果摘要
    classifier.show_results_summary()
    
    # 分析SVM性能
    classifier.analyze_svm_performance()
    
    # 自动导出ONNX模型
    print("\n开始导出ONNX模型...")
    classifier.export_all_models_to_onnx()
    
    print("\n所有任务完成！")

if __name__ == "__main__":
    main()
