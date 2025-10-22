#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简化版SVM训练脚本 - 快速运行版本
"""

from svm_four_class_trainer import SVMEmotionClassifier

def quick_train():
    """
    快速训练和评估
    """
    print("开始快速SVM训练...")
    
    # 初始化分类器
    classifier = SVMEmotionClassifier('svm/3_processed.csv')
    
    # 加载和预处理数据
    classifier.load_and_preprocess_data(test_size=0.2, random_state=42)
    
    # 使用默认参数训练（可以根据需要调整）
    classifier.train_model(C=1000, gamma='scale', kernel='rbf')
    
    # 评估模型
    results = classifier.evaluate_model()
    
    # 保存模型
    classifier.save_model('svm_emotion_model.joblib')
    
    print(f"\n训练完成！最终准确率: {results['accuracy']:.4f}")
    
    return classifier

if __name__ == "__main__":
    classifier = quick_train()
