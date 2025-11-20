#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SVM四分类训练脚本
数据集：HRV特征数据，四种情绪分类（开心、悲伤、平静、焦虑）
包含完整的模型评估环节
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import (classification_report, confusion_matrix, 
                           accuracy_score, precision_score, recall_score, 
                           f1_score, roc_auc_score, roc_curve)
from sklearn.multiclass import OneVsRestClassifier
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

# ONNX相关导入
try:
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType
    import onnx
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    print("警告: ONNX相关包未安装，将无法导出ONNX模型")
    print("如需使用ONNX导出，请运行: pip install skl2onnx onnx onnxruntime")

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class SVMEmotionClassifier:
    def __init__(self, data_path):
        """
        初始化SVM情绪分类器
        
        Args:
            data_path (str): 数据文件路径
        """
        self.data_path = data_path
        self.model = None
        self.scaler = None
        self.label_encoder = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.feature_names = None
        
    def load_and_preprocess_data(self, test_size=0.2, random_state=42):
        """
        加载和预处理数据
        
        Args:
            test_size (float): 测试集比例
            random_state (int): 随机种子
        """
        print("正在加载数据...")
        
        # 读取数据
        df = pd.read_csv(self.data_path)
        
        # 显示数据基本信息
        print(f"数据形状: {df.shape}")
        print(f"情绪类别分布:")
        print(df['emotion'].value_counts())
        print(f"\n缺失值统计:")
        print(df.isnull().sum())
        
        # 分离特征和标签
        X = df.drop('emotion', axis=1)
        y = df['emotion']
        
        self.feature_names = X.columns.tolist()
        
        # 标签编码
        self.label_encoder = LabelEncoder()
        y_encoded = self.label_encoder.fit_transform(y)
        
        print(f"\n标签编码映射:")
        for i, label in enumerate(self.label_encoder.classes_):
            print(f"{i}: {label}")
        
        # 分割训练集和测试集
        X_train_df, X_test_df, y_train, y_test = train_test_split(
            X, y_encoded, test_size=test_size, random_state=random_state, 
            stratify=y_encoded
        )
        
        # 确保测试集包含所有类别
        unique_classes = np.unique(y_encoded)
        y_train = np.array(y_train)
        y_test = np.array(y_test)
        test_classes = np.unique(y_test)
        missing_classes = set(unique_classes) - set(test_classes)
        
        if missing_classes:
            print(f"\n警告: 测试集中缺少以下类别: {[self.label_encoder.classes_[c] for c in missing_classes]}")
            print("正在从训练集中移动样本到测试集...")
            
            X_train_df = X_train_df.reset_index(drop=True)
            X_test_df = X_test_df.reset_index(drop=True)
            
            for class_label in missing_classes:
                train_class_indices = np.where(y_train == class_label)[0]
                if len(train_class_indices) > 0:
                    move_idx = train_class_indices[0]
                    X_test_df = pd.concat([X_test_df, X_train_df.iloc[[move_idx]]], ignore_index=True)
                    y_test = np.append(y_test, y_train[move_idx])
                    X_train_df = X_train_df.drop(X_train_df.index[move_idx]).reset_index(drop=True)
                    y_train = np.delete(y_train, move_idx)
                    print(f"  已移动类别 {self.label_encoder.classes_[class_label]} 的样本到测试集")
        
        # 转换为numpy数组用于后续处理
        self.X_train = X_train_df.values
        self.X_test = X_test_df.values
        self.y_train = y_train
        self.y_test = y_test
        
        # 验证并显示测试集类别分布
        final_test_classes = np.unique(self.y_test)
        if len(final_test_classes) == len(unique_classes):
            print(f"\n✓ 测试集已包含所有 {len(unique_classes)} 个类别")
        else:
            print(f"\n警告: 测试集仍缺少某些类别")
        print(f"测试集中各类别样本数:")
        for cls in unique_classes:
            count = np.sum(self.y_test == cls)
            print(f"  {self.label_encoder.classes_[cls]}: {count}")
        
        # 特征标准化
        self.scaler = StandardScaler()
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)
        
        print(f"\n训练集形状: {self.X_train.shape}")
        print(f"测试集形状: {self.X_test.shape}")
        
    def hyperparameter_tuning(self, cv=5):
        """
        超参数调优
        
        Args:
            cv (int): 交叉验证折数
        """
        print("\n开始超参数调优...")
        
        # 定义参数网格
        param_grid = {
            'C': [0.1, 1, 10, 100],
            'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1],
            'kernel': ['rbf', 'poly', 'sigmoid']
        }
        
        # 网格搜索
        svm = SVC(random_state=42, probability=True)
        grid_search = GridSearchCV(
            svm, param_grid, cv=cv, scoring='accuracy', 
            n_jobs=-1, verbose=1
        )
        
        grid_search.fit(self.X_train, self.y_train)
        
        print(f"最佳参数: {grid_search.best_params_}")
        print(f"最佳交叉验证分数: {grid_search.best_score_:.4f}")
        
        self.model = grid_search.best_estimator_
        
        return grid_search.best_params_
    
    def train_model(self, **kwargs):
        """
        训练SVM模型
        
        Args:
            **kwargs: SVM参数
        """
        if self.model is None:
            print("\n使用默认参数训练模型...")
            default_params = {'C': 10, 'gamma': 'scale', 'kernel': 'rbf'}
            default_params.update(kwargs)
            self.model = SVC(**default_params, random_state=42, probability=True)
        
        print("正在训练模型...")
        self.model.fit(self.X_train, self.y_train)
        print("模型训练完成！")
    
    def evaluate_model(self):
        """
        评估模型性能
        """
        print("\n" + "="*50)
        print("模型评估结果")
        print("="*50)
        
        # 预测
        y_pred = self.model.predict(self.X_test)
        y_pred_proba = self.model.predict_proba(self.X_test)
        
        # 基本指标
        accuracy = accuracy_score(self.y_test, y_pred)
        precision = precision_score(self.y_test, y_pred, average='weighted')
        recall = recall_score(self.y_test, y_pred, average='weighted')
        f1 = f1_score(self.y_test, y_pred, average='weighted')
        
        print(f"准确率 (Accuracy): {accuracy:.4f}")
        print(f"精确率 (Precision): {precision:.4f}")
        print(f"召回率 (Recall): {recall:.4f}")
        print(f"F1分数: {f1:.4f}")
        
        # 详细分类报告
        print(f"\n详细分类报告:")
        target_names = self.label_encoder.classes_
        print(classification_report(self.y_test, y_pred, 
                                  target_names=target_names))
        
        # 混淆矩阵
        self.plot_confusion_matrix(self.y_test, y_pred, target_names)
        
        # 交叉验证
        self.cross_validation_evaluation()
        
        # ROC曲线（多分类）
        self.plot_multiclass_roc_curve(y_pred_proba)
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }
    
    def plot_confusion_matrix(self, y_true, y_pred, target_names):
        """
        绘制混淆矩阵
        """
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=target_names, yticklabels=target_names)
        plt.title('混淆矩阵 (Confusion Matrix)')
        plt.xlabel('预测标签')
        plt.ylabel('真实标签')
        plt.tight_layout()
        plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 计算每个类别的准确率
        print(f"\n各类别分类准确率:")
        for i, class_name in enumerate(target_names):
            class_accuracy = cm[i, i] / np.sum(cm[i, :])
            print(f"{class_name}: {class_accuracy:.4f}")
    
    def cross_validation_evaluation(self, cv=5):
        """
        交叉验证评估
        """
        print(f"\n{cv}折交叉验证结果:")
        
        # 重新组合训练数据进行交叉验证
        X_full = np.vstack([self.X_train, self.X_test])
        y_full = np.hstack([self.y_train, self.y_test])
        
        cv_scores = cross_val_score(self.model, X_full, y_full, cv=cv, scoring='accuracy')
        
        print(f"交叉验证分数: {cv_scores}")
        print(f"平均准确率: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    def plot_multiclass_roc_curve(self, y_pred_proba):
        """
        绘制多分类ROC曲线
        """
        from sklearn.preprocessing import label_binarize
        from sklearn.metrics import roc_curve, auc
        from itertools import cycle
        
        # 二值化标签
        y_test_bin = label_binarize(self.y_test, classes=range(len(self.label_encoder.classes_)))
        n_classes = y_test_bin.shape[1]
        
        # 计算每个类别的ROC曲线和AUC
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_pred_proba[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
        
        # 绘制ROC曲线
        plt.figure(figsize=(12, 8))
        colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'red'])
        
        for i, color in zip(range(n_classes), colors):
            plt.plot(fpr[i], tpr[i], color=color, lw=2,
                    label=f'{self.label_encoder.classes_[i]} (AUC = {roc_auc[i]:.2f})')
        
        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('假正率 (False Positive Rate)')
        plt.ylabel('真正率 (True Positive Rate)')
        plt.title('多分类ROC曲线')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('multiclass_roc_curve.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 打印AUC分数
        print(f"\n各类别AUC分数:")
        for i, class_name in enumerate(self.label_encoder.classes_):
            print(f"{class_name}: {roc_auc[i]:.4f}")
    
    def feature_importance_analysis(self):
        """
        特征重要性分析（基于线性SVM的权重）
        """
        if self.model.kernel != 'linear':
            print("\n注意: 当前模型使用非线性核函数，无法直接分析特征重要性")
            return
        
        # 获取特征权重
        weights = np.abs(self.model.coef_[0])
        
        # 创建特征重要性DataFrame
        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': weights
        }).sort_values('importance', ascending=False)
        
        print(f"\n特征重要性排序:")
        print(feature_importance)
        
        # 绘制特征重要性图
        plt.figure(figsize=(10, 6))
        sns.barplot(data=feature_importance, x='importance', y='feature')
        plt.title('特征重要性分析')
        plt.xlabel('重要性权重')
        plt.tight_layout()
        plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def save_model(self, model_path='svm_emotion_classifier.onnx', save_onnx=True):
        """
        保存训练好的模型为ONNX格式
        
        Args:
            model_path (str): 模型保存路径（.onnx文件）
            save_onnx (bool): 是否保存ONNX格式，False则保存为joblib格式
        """
        if save_onnx and ONNX_AVAILABLE:
            try:
                # 创建包含scaler和model的Pipeline
                pipeline = Pipeline([
                    ('scaler', self.scaler),
                    ('svm', self.model)
                ])
                
                # 获取特征数量
                n_features = len(self.feature_names)
                
                # 定义输入类型
                initial_type = [('float_input', FloatTensorType([None, n_features]))]
                
                # 转换为ONNX格式
                onnx_model = convert_sklearn(
                    pipeline,
                    initial_types=initial_type,
                    target_opset=11
                )
                
                # 保存ONNX模型
                with open(model_path, "wb") as f:
                    f.write(onnx_model.SerializeToString())
                
                print(f"\nONNX模型已保存至: {model_path}")
                
                # 保存label_encoder和特征名称信息
                info_path = model_path.replace('.onnx', '_info.joblib')
                model_info = {
                    'label_encoder': self.label_encoder,
                    'feature_names': self.feature_names,
                    'n_features': n_features,
                    'classes': self.label_encoder.classes_.tolist()
                }
                joblib.dump(model_info, info_path)
                print(f"模型信息已保存至: {info_path}")
                
                # 打印模型信息
                print(f"\n模型信息:")
                print(f"  特征数量: {n_features}")
                print(f"  特征名称: {', '.join(self.feature_names)}")
                print(f"  类别数量: {len(self.label_encoder.classes_)}")
                print(f"  类别标签: {', '.join(self.label_encoder.classes_)}")
                
            except Exception as e:
                print(f"\n错误: 导出ONNX模型失败: {e}")
                print("将改用joblib格式保存")
                joblib_path = model_path.replace('.onnx', '.joblib')
                model_data = {
                    'model': self.model,
                    'scaler': self.scaler,
                    'label_encoder': self.label_encoder,
                    'feature_names': self.feature_names
                }
                joblib.dump(model_data, joblib_path)
                print(f"模型已保存至: {joblib_path}")
            
        elif save_onnx and not ONNX_AVAILABLE:
            print("\n警告: ONNX功能不可用，改用joblib格式保存")
            joblib_path = model_path.replace('.onnx', '.joblib')
            model_data = {
                'model': self.model,
                'scaler': self.scaler,
                'label_encoder': self.label_encoder,
                'feature_names': self.feature_names
            }
            joblib.dump(model_data, joblib_path)
            print(f"模型已保存至: {joblib_path}")
        else:
            # 保存为joblib格式
            if not model_path.endswith('.joblib'):
                model_path = model_path.replace('.onnx', '.joblib')
            model_data = {
                'model': self.model,
                'scaler': self.scaler,
                'label_encoder': self.label_encoder,
                'feature_names': self.feature_names
            }
            joblib.dump(model_data, model_path)
            print(f"\n模型已保存至: {model_path}")
    
    def load_model(self, model_path='svm_emotion_classifier.joblib', load_onnx=False):
        """
        加载训练好的模型
        
        Args:
            model_path (str): 模型文件路径
            load_onnx (bool): 是否加载ONNX格式模型
        """
        if load_onnx and model_path.endswith('.onnx'):
            try:
                import onnxruntime as ort
                
                # 加载ONNX模型
                self.onnx_session = ort.InferenceSession(model_path)
                self.input_name = self.onnx_session.get_inputs()[0].name
                self.output_name = self.onnx_session.get_outputs()[0].name
                
                # 加载模型信息
                info_path = model_path.replace('.onnx', '_info.joblib')
                if os.path.exists(info_path):
                    model_info = joblib.load(info_path)
                    self.feature_names = model_info['feature_names']
                    
                    # 恢复label_encoder
                    try:
                        self.label_encoder = model_info['label_encoder']
                        # 验证label_encoder是否有效
                        _ = self.label_encoder.classes_
                    except:
                        # 如果无法恢复，使用备份的classes列表重建
                        le = LabelEncoder()
                        le.classes_ = np.array(model_info['classes'])
                        self.label_encoder = le
                    
                    print(f"ONNX模型已从 {model_path} 加载")
                    print(f"模型信息: {len(self.feature_names)} 个特征, {len(self.label_encoder.classes_)} 个类别")
                else:
                    raise FileNotFoundError(f"模型信息文件不存在: {info_path}")
                    
            except ImportError:
                print("错误: 需要安装onnxruntime来加载ONNX模型")
                print("请运行: pip install onnxruntime")
                raise
            except Exception as e:
                print(f"加载ONNX模型失败: {e}")
                raise
        else:
            # 加载joblib格式
            if not model_path.endswith('.joblib'):
                # 尝试查找joblib文件
                joblib_path = model_path.replace('.onnx', '.joblib')
                if os.path.exists(joblib_path):
                    model_path = joblib_path
            
            model_data = joblib.load(model_path)
            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.label_encoder = model_data['label_encoder']
            self.feature_names = model_data['feature_names']
            print(f"模型已从 {model_path} 加载")
    
    def predict_new_sample(self, sample):
        """
        预测新样本
        
        Args:
            sample: 新样本特征（list或array）
        """
        # 检查是否使用ONNX模型
        if hasattr(self, 'onnx_session'):
            return self._predict_with_onnx(sample)
        
        if self.model is None:
            raise ValueError("模型未训练，请先训练模型")
        
        # 确保输入是正确的格式
        if isinstance(sample, list):
            sample = np.array(sample).reshape(1, -1)
        elif len(sample.shape) == 1:
            sample = sample.reshape(1, -1)
        
        # 标准化
        sample_scaled = self.scaler.transform(sample)
        
        # 预测
        prediction = self.model.predict(sample_scaled)[0]
        probability = self.model.predict_proba(sample_scaled)[0]
        
        # 解码标签
        emotion = self.label_encoder.inverse_transform([prediction])[0]
        
        # 显示结果
        print(f"\n预测结果: {emotion}")
        print(f"各类别概率:")
        for i, class_name in enumerate(self.label_encoder.classes_):
            print(f"  {class_name}: {probability[i]:.4f}")
        
        return emotion, probability
    
    def _predict_with_onnx(self, sample):
        """
        使用ONNX模型进行预测
        """
        import onnxruntime as ort
        
        # 确保输入是正确的格式
        if isinstance(sample, list):
            sample = np.array(sample, dtype=np.float32).reshape(1, -1)
        elif len(sample.shape) == 1:
            sample = sample.reshape(1, -1).astype(np.float32)
        else:
            sample = sample.astype(np.float32)
        
        # 使用ONNX模型预测
        inputs = {self.input_name: sample}
        outputs = self.onnx_session.run([self.output_name], inputs)
        
        # 获取预测结果（ONNX模型输出概率）
        probabilities = outputs[0][0]
        prediction = np.argmax(probabilities)
        
        # 解码标签
        emotion = self.label_encoder.inverse_transform([prediction])[0]
        
        # 显示结果
        print(f"\n预测结果: {emotion}")
        print(f"各类别概率:")
        for i, class_name in enumerate(self.label_encoder.classes_):
            print(f"  {class_name}: {probabilities[i]:.4f}")
        
        return emotion, probabilities


def main():
    """
    主函数
    """
    print("SVM四分类情绪识别系统")
    print("="*50)
    
    # 初始化分类器
    classifier = SVMEmotionClassifier(r'C:\Users\QAQ\Desktop\emotion\hrv_FB_train.csv')
    
    # 加载和预处理数据
    classifier.load_and_preprocess_data(test_size=0.2, random_state=42)
    
    # 选择是否进行超参数调优
    use_tuning = input("\n是否进行超参数调优? (y/n, 默认n): ").lower().strip()
    
    if use_tuning == 'y':
        # 超参数调优
        best_params = classifier.hyperparameter_tuning(cv=5)
    else:
        # 使用默认参数训练
        classifier.train_model()
    
    # 评估模型
    results = classifier.evaluate_model()
    
    # 特征重要性分析（如果使用线性核）
    if hasattr(classifier.model, 'kernel') and classifier.model.kernel == 'linear':
        classifier.feature_importance_analysis()
    
    # 保存模型
    save_model = input("\n是否保存模型? (y/n, 默认y): ").lower().strip()
    if save_model != 'n':
        classifier.save_model(model_path='svm_emotion_classifier.onnx', save_onnx=True)
    
    # 演示预测新样本
    print("\n" + "="*50)
    print("演示: 预测新样本")
    print("="*50)
    
    # 使用测试集中的一个样本作为演示
    sample_idx = 0
    sample_features = classifier.X_test[sample_idx]
    true_label = classifier.label_encoder.inverse_transform([classifier.y_test[sample_idx]])[0]
    
    print(f"真实标签: {true_label}")
    classifier.predict_new_sample(sample_features)


if __name__ == "__main__":
    main()

