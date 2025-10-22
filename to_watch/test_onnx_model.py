#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ONNX模型测试脚本
用于加载和测试情绪分类的ONNX模型
"""

import numpy as np
import onnxruntime as ort
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
try:
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
import os
import sys

class ONNXEmotionClassifier:
    def __init__(self, onnx_model_path, model_info_path=None):
        """
        初始化ONNX情绪分类器
        
        Args:
            onnx_model_path: ONNX模型文件路径
            model_info_path: 模型信息文件路径（可选）
        """
        self.onnx_model_path = onnx_model_path
        self.model_info_path = model_info_path
        self.session = None
        self.input_name = None
        self.output_name = None
        self.scaler = None
        self.emotions = []
        self.feature_names = ['RMSSD', 'pNN58', 'SDNN', 'SD1', 'SD2', 'SD1_SD2']
        
        self.load_model()
        self.load_model_info()
    
    def load_model(self):
        """加载ONNX模型"""
        try:
            if not os.path.exists(self.onnx_model_path):
                raise FileNotFoundError(f"ONNX模型文件不存在: {self.onnx_model_path}")
            
            # 创建ONNX Runtime会话
            self.session = ort.InferenceSession(self.onnx_model_path)
            
            # 获取输入输出信息
            self.input_name = self.session.get_inputs()[0].name
            self.output_name = self.session.get_outputs()[0].name
            
            print(f"✓ ONNX模型加载成功: {self.onnx_model_path}")
            print(f"  输入名称: {self.input_name}")
            print(f"  输出名称: {self.output_name}")
            print(f"  输入形状: {self.session.get_inputs()[0].shape}")
            print(f"  输出形状: {self.session.get_outputs()[0].shape}")
            
        except Exception as e:
            print(f"✗ ONNX模型加载失败: {e}")
            sys.exit(1)
    
    def load_model_info(self):
        """加载模型信息"""
        if self.model_info_path and os.path.exists(self.model_info_path):
            try:
                with open(self.model_info_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    for line in lines:
                        if line.startswith('情绪标签:'):
                            emotions_str = line.split(':', 1)[1].strip()
                            # 解析情绪标签，格式可能是 ['平静', '悲伤'] 或 平静,悲伤
                            if emotions_str.startswith('[') and emotions_str.endswith(']'):
                                emotions_str = emotions_str[1:-1]
                            self.emotions = [e.strip().strip("'\"") for e in emotions_str.split(',')]
                            break
                
                print(f"✓ 模型信息加载成功: {self.emotions}")
            except Exception as e:
                print(f"⚠ 模型信息加载失败: {e}")
        
        # 如果没有从文件加载到情绪标签，尝试从模型名称推断
        if not self.emotions:
            model_name = os.path.basename(self.onnx_model_path)
            if '平静' in model_name and '悲伤' in model_name:
                self.emotions = ['平静', '悲伤']
                print(f"✓ 从模型名称推断情绪标签: {self.emotions}")
    
    def predict(self, features):
        """
        使用ONNX模型进行预测
        
        Args:
            features: 特征数组，形状为 (n_samples, 6) 或 (6,)
                    特征顺序: [RMSSD, pNN58, SDNN, SD1, SD2, SD1_SD2]
        
        Returns:
            predictions: 预测结果
            probabilities: 预测概率（如果模型支持）
        """
        try:
            # 确保输入是numpy数组
            features = np.array(features, dtype=np.float32)
            
            # 如果是单个样本，添加batch维度
            if features.ndim == 1:
                features = features.reshape(1, -1)
            
            # 检查特征数量
            if features.shape[1] != 6:
                raise ValueError(f"期望6个特征，但得到{features.shape[1]}个")
            
            # 使用ONNX模型进行推理
            inputs = {self.input_name: features}
            outputs = self.session.run([self.output_name], inputs)
            
            predictions = outputs[0]
            
            # 如果有情绪标签，将数字预测转换为标签
            if self.emotions and len(self.emotions) == 2:
                # 对于二分类，通常0对应第一个情绪，1对应第二个情绪
                label_predictions = [self.emotions[pred] for pred in predictions]
                return label_predictions, predictions
            else:
                return predictions, predictions
                
        except Exception as e:
            print(f"✗ 预测失败: {e}")
            return None, None
    
    def predict_with_probabilities(self, features):
        """
        预测并获取概率（如果模型支持）
        """
        try:
            features = np.array(features, dtype=np.float32)
            if features.ndim == 1:
                features = features.reshape(1, -1)
            
            # 获取所有输出
            inputs = {self.input_name: features}
            outputs = self.session.run(None, inputs)
            
            # 通常第一个输出是预测，第二个是概率
            predictions = outputs[0]
            probabilities = outputs[1] if len(outputs) > 1 else None
            
            if self.emotions and len(self.emotions) == 2:
                label_predictions = [self.emotions[pred] for pred in predictions]
                return label_predictions, predictions, probabilities
            else:
                return predictions, predictions, probabilities
                
        except Exception as e:
            print(f"✗ 带概率预测失败: {e}")
            return None, None, None

def create_sample_data():
    """创建示例测试数据"""
    print("\n创建示例测试数据...")
    
    # 基于真实HRV数据的合理范围创建示例数据
    sample_data = [
        # 平静状态的特征值
        [45.2, 12.8, 52.3, 23.1, 29.2, 0.79],  # 平静样本1
        [38.7, 9.5, 48.6, 20.4, 28.2, 0.72],   # 平静样本2
        [42.1, 11.2, 51.0, 22.3, 28.7, 0.78],  # 平静样本3
        
        # 悲伤状态的特征值
        [28.4, 5.2, 38.9, 15.8, 23.1, 0.68],   # 悲伤样本1
        [31.6, 6.8, 42.3, 17.2, 25.1, 0.69],   # 悲伤样本2
        [26.9, 4.9, 36.7, 14.9, 21.8, 0.68],   # 悲伤样本3
    ]
    
    sample_labels = ['平静', '平静', '平静', '悲伤', '悲伤', '悲伤']
    
    return np.array(sample_data), sample_labels

def test_model_performance(classifier, test_data, test_labels):
    """测试模型性能"""
    print(f"\n测试模型性能...")
    print(f"测试样本数: {len(test_data)}")
    
    predictions, raw_predictions, probabilities = classifier.predict_with_probabilities(test_data)
    
    if predictions is None:
        print("✗ 预测失败，无法评估性能")
        return
    
    # 计算准确率
    correct = sum(1 for pred, true in zip(predictions, test_labels) if pred == true)
    accuracy = correct / len(test_labels)
    
    print(f"✓ 预测完成")
    print(f"准确率: {accuracy:.2%} ({correct}/{len(test_labels)})")
    
    # 显示详细结果
    print(f"\n详细预测结果:")
    print(f"{'样本':<6} {'真实标签':<8} {'预测标签':<8} {'原始预测':<8} {'概率(如果有)':<15}")
    print("-" * 60)
    
    for i, (true_label, pred_label, raw_pred) in enumerate(zip(test_labels, predictions, raw_predictions)):
        prob_str = ""
        if probabilities is not None:
            prob = probabilities[i][raw_pred]
            prob_str = f"{prob:.3f}"
        
        print(f"{i+1:<6} {true_label:<8} {pred_label:<8} {raw_pred:<8} {prob_str:<15}")

def main():
    """主函数"""
    print("ONNX情绪分类模型测试")
    print("=" * 50)
    
    # 模型路径
    onnx_model_path = "emotion_onnx_models/平静_vs_悲伤.onnx"
    model_info_path = "emotion_onnx_models/平静_vs_悲伤_info.txt"
    
    try:
        # 初始化分类器
        print("初始化ONNX分类器...")
        classifier = ONNXEmotionClassifier(onnx_model_path, model_info_path)
        
        # 创建测试数据
        test_data, test_labels = create_sample_data()
        
        # 单个样本测试
        print(f"\n单个样本测试:")
        single_sample = test_data[0]
        print(f"输入特征: {single_sample}")
        print(f"特征名称: {classifier.feature_names}")
        
        prediction, raw_pred, probability = classifier.predict_with_probabilities(single_sample)
        print(f"预测结果: {prediction[0] if prediction else 'None'}")
        print(f"原始预测: {raw_pred[0] if raw_pred is not None else 'None'}")
        if probability is not None:
            print(f"预测概率: {probability[0]}")
        
        # 批量测试
        print(f"\n批量样本测试:")
        predictions, raw_predictions, probabilities = classifier.predict_with_probabilities(test_data)
        
        if predictions:
            print(f"批量预测结果: {predictions}")
        
        # 性能测试
        test_model_performance(classifier, test_data, test_labels)
        
        # 交互式测试
        print(f"\n交互式测试:")
        print("请输入HRV特征值进行测试 (6个值，用空格分隔):")
        print("特征顺序: RMSSD, pNN58, SDNN, SD1, SD2, SD1_SD2")
        print("示例: 45.2 12.8 52.3 23.1 29.2 0.79")
        print("输入 'quit' 退出")
        
        while True:
            user_input = input("\n请输入特征值: ").strip()
            if user_input.lower() == 'quit':
                break
            
            try:
                features = [float(x) for x in user_input.split()]
                if len(features) != 6:
                    print("✗ 请输入6个特征值")
                    continue
                
                prediction, raw_pred, probability = classifier.predict_with_probabilities(features)
                print(f"预测结果: {prediction[0] if prediction else 'None'}")
                if probability is not None:
                    prob = probability[0][raw_pred[0]]
                    print(f"预测概率: {prob:.3f}")
                    
            except ValueError:
                print("✗ 输入格式错误，请输入6个数字")
            except Exception as e:
                print(f"✗ 预测失败: {e}")
        
        print("\n测试完成！")
        
    except FileNotFoundError as e:
        print(f"✗ 文件未找到: {e}")
        print("请确保ONNX模型文件存在")
    except Exception as e:
        print(f"✗ 测试过程中出现错误: {e}")

if __name__ == "__main__":
    main()
