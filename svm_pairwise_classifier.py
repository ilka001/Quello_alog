import csv
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import itertools
from collections import Counter
import warnings
import os
try:
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType
    import onnx
    ONNX_AVAILABLE = True
except ImportError:
    print("警告: ONNX相关包未安装，将跳过ONNX导出功能")
    print("如需使用ONNX导出，请运行: pip install skl2onnx onnx")
    ONNX_AVAILABLE = False
warnings.filterwarnings('ignore')

class SVMPairwiseClassifier:
    def __init__(self, data_file):
        self.data_file = data_file
        self.data = []
        self.emotions = []
        self.scaler = StandardScaler()
        self.models = {}
        self.results = {}
        self.onnx_models_dir = "emotion_onnx_models"
        
    def load_data(self):
        """加载CSV数据"""
        print("正在加载数据...")
        with open(self.data_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # 提取特征
                features = [
                    float(row['RMSSD']),
                    float(row['pNN58']),
                    float(row['SDNN']),
                    float(row['SD1']),
                    float(row['SD2']),
                    float(row['SD1_SD2']),
                ]
                emotion = row['emotion']
                self.data.append(features + [emotion])
                if emotion not in self.emotions:
                    self.emotions.append(emotion)
        
        print(f"数据加载完成，共 {len(self.data)} 条记录")
        print(f"情绪标签: {self.emotions}")
        
        # 统计各情绪的数量
        emotion_counts = {}
        for row in self.data:
            emotion = row[-1]
            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
        
        print("情绪分布:")
        for emotion, count in emotion_counts.items():
            print(f"  {emotion}: {count} 条")
    
    def get_emotion_pairs(self):
        """获取所有可能的两两组合"""
        pairs = list(itertools.combinations(self.emotions, 2))
        print(f"\n将训练 {len(pairs)} 个SVM两两分类器:")
        for i, (emotion1, emotion2) in enumerate(pairs, 1):
            print(f"{i}. {emotion1} vs {emotion2}")
        return pairs
    
    def prepare_pair_data(self, emotion1, emotion2):
        """准备指定情绪对的数据"""
        X = []
        y = []
        
        for row in self.data:
            if row[-1] in [emotion1, emotion2]:
                X.append(row[:-1])  # 特征
                y.append(row[-1])   # 标签
        
        X = np.array(X)
        y = np.array(y)
        
        print(f"\n{emotion1} vs {emotion2} 数据统计:")
        print(f"总样本数: {len(y)}")
        print(f"{emotion1}: {np.sum(y == emotion1)} 个样本")
        print(f"{emotion2}: {np.sum(y == emotion2)} 个样本")
        
        return X, y
    
    def train_svm_pair(self, emotion1, emotion2, test_size=0.2, random_state=42):
        """训练SVM两两分类器"""
        print(f"\n开始训练SVM {emotion1} vs {emotion2} 分类器...")
        
        # 准备数据
        X, y = self.prepare_pair_data(emotion1, emotion2)
        
        if len(X) < 10:
            print(f"警告: 数据量太少 ({len(X)} 个样本)，跳过训练")
            return None
        
        # 分割训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # 标准化特征
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # 网格搜索最佳参数
        param_grid = {
            'C': [0.1, 1, 10, 100],
            'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1],
            'kernel': ['rbf', 'linear', 'poly']
        }
        
        svm = SVC(random_state=random_state)
        grid_search = GridSearchCV(
            svm, param_grid, cv=5, scoring='accuracy', n_jobs=-1
        )
        
        print("正在进行网格搜索优化SVM参数...")
        grid_search.fit(X_train_scaled, y_train)
        
        # 获取最佳模型
        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_
        
        print(f"最佳SVM参数: {best_params}")
        print(f"最佳交叉验证分数: {grid_search.best_score_:.4f}")
        
        # 在测试集上评估
        y_pred = best_model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"测试集准确率: {accuracy:.4f}")
        print("\nSVM分类报告:")
        print(classification_report(y_test, y_pred))
        
        # 显示混淆矩阵
        cm = confusion_matrix(y_test, y_pred)
        print("混淆矩阵:")
        print(f"实际\\预测  {emotion1:>8}  {emotion2:>8}")
        print(f"{emotion1:>8}  {cm[0,0]:>8}  {cm[0,1]:>8}")
        print(f"{emotion2:>8}  {cm[1,0]:>8}  {cm[1,1]:>8}")
        
        # 保存模型和结果
        pair_name = f"{emotion1}_vs_{emotion2}"
        self.models[pair_name] = {
            'model': best_model,
            'scaler': self.scaler,
            'emotions': [emotion1, emotion2]
        }
        
        self.results[pair_name] = {
            'accuracy': accuracy,
            'best_params': best_params,
            'cv_score': grid_search.best_score_,
            'test_size': len(X_test),
            'train_size': len(X_train),
            'confusion_matrix': cm
        }
        
        return best_model, accuracy
    
    def train_all_svm_pairs(self):
        """训练所有SVM两两组合"""
        pairs = self.get_emotion_pairs()
        
        print("\n" + "="*80)
        
        for emotion1, emotion2 in pairs:
            try:
                self.train_svm_pair(emotion1, emotion2)
                print("="*80)
            except Exception as e:
                print(f"训练SVM {emotion1} vs {emotion2} 时出错: {e}")
                print("="*80)
    
    def predict_emotion(self, features, emotion1, emotion2):
        """使用训练好的SVM模型预测情绪"""
        pair_name = f"{emotion1}_vs_{emotion2}"
        if pair_name not in self.models:
            raise ValueError(f"SVM模型 {pair_name} 未训练")
        
        model_info = self.models[pair_name]
        features_scaled = model_info['scaler'].transform([features])
        prediction = model_info['model'].predict(features_scaled)[0]
        
        return prediction
    
    def show_results_summary(self):
        """显示所有SVM结果摘要"""
        if not self.results:
            print("没有SVM训练结果")
            return
        
        print("\n所有SVM两两分类结果摘要:")
        print("-" * 100)
        print(f"{'SVM分类器':<25} {'准确率':<10} {'交叉验证分数':<15} {'最佳核函数':<15} {'训练样本':<10} {'测试样本':<10}")
        print("-" * 100)
        
        for pair_name, result in self.results.items():
            kernel = result['best_params']['kernel']
            print(f"{pair_name:<25} {result['accuracy']:<10.4f} {result['cv_score']:<15.4f} "
                  f"{kernel:<15} {result['train_size']:<10} {result['test_size']:<10}")
        
        # 找出最佳SVM分类器
        if self.results:
            best_pair = max(self.results.items(), key=lambda x: x[1]['accuracy'])
            print(f"\n最佳SVM分类器: {best_pair[0]}")
            print(f"准确率: {best_pair[1]['accuracy']:.4f}")
            print(f"最佳参数: {best_pair[1]['best_params']}")
            
            # 显示所有结果排序
            sorted_results = sorted(self.results.items(), key=lambda x: x[1]['accuracy'], reverse=True)
            print(f"\n所有SVM分类器按准确率排序:")
            for i, (pair_name, result) in enumerate(sorted_results, 1):
                kernel = result['best_params']['kernel']
                print(f"{i:2d}. {pair_name:<25} {result['accuracy']:.4f} (核函数: {kernel})")
    
    def analyze_svm_performance(self):
        """分析SVM性能"""
        if not self.results:
            print("没有结果可分析")
            return
        
        print("\nSVM性能分析:")
        print("-" * 50)
        
        # 统计不同核函数的效果
        kernel_performance = {}
        for pair_name, result in self.results.items():
            kernel = result['best_params']['kernel']
            if kernel not in kernel_performance:
                kernel_performance[kernel] = []
            kernel_performance[kernel].append(result['accuracy'])
        
        print("不同核函数的平均性能:")
        for kernel, accuracies in kernel_performance.items():
            avg_acc = np.mean(accuracies)
            print(f"  {kernel}: {avg_acc:.4f} (基于{len(accuracies)}个分类器)")
        
        # 分析参数分布
        c_values = [result['best_params']['C'] for result in self.results.values()]
        gamma_values = [result['best_params']['gamma'] for result in self.results.values()]
        
        print(f"\n参数分布:")
        print(f"  C值分布: {Counter(c_values)}")
        print(f"  gamma值分布: {Counter(gamma_values)}")
    
    def create_onnx_dir(self):
        """创建ONNX模型目录"""
        if not os.path.exists(self.onnx_models_dir):
            os.makedirs(self.onnx_models_dir)
            print(f"已创建ONNX模型目录: {self.onnx_models_dir}")
    
    def export_model_to_onnx(self, pair_name, model_info):
        """将单个SVM模型导出为ONNX格式"""
        try:
            model = model_info['model']
            scaler = model_info['scaler']
            emotions = model_info['emotions']
            
            # 定义输入类型（6个特征）
            initial_type = [('float_input', FloatTensorType([None, 6]))]
            
            # 转换模型为ONNX格式
            onnx_model = convert_sklearn(
                model,
                initial_types=initial_type,
                target_opset=11
            )
            
            # 保存ONNX模型
            onnx_filename = f"{pair_name}.onnx"
            onnx_path = os.path.join(self.onnx_models_dir, onnx_filename)
            
            with open(onnx_path, "wb") as f:
                f.write(onnx_model.SerializeToString())
            
            print(f"已导出ONNX模型: {onnx_path}")
            
            # 同时保存scaler和emotions信息到文本文件
            info_filename = f"{pair_name}_info.txt"
            info_path = os.path.join(self.onnx_models_dir, info_filename)
            
            with open(info_path, 'w', encoding='utf-8') as f:
                f.write(f"模型名称: {pair_name}\n")
                f.write(f"情绪标签: {emotions}\n")
                f.write(f"特征数量: 6\n")
                f.write(f"特征顺序: RMSSD, pNN58, SDNN, SD1, SD2, SD1_SD2\n")
                f.write(f"标准化器: StandardScaler\n")
                f.write(f"模型类型: SVM\n")
            
            return True
            
        except Exception as e:
            print(f"导出模型 {pair_name} 到ONNX时出错: {e}")
            return False
    
    def export_all_models_to_onnx(self):
        """导出所有训练好的模型为ONNX格式"""
        if not ONNX_AVAILABLE:
            print("ONNX功能不可用，跳过模型导出")
            return
            
        if not self.models:
            print("没有训练好的模型可以导出")
            return
        
        print(f"\n开始导出所有模型到ONNX格式...")
        print(f"导出目录: {self.onnx_models_dir}")
        
        # 创建ONNX目录
        self.create_onnx_dir()
        
        success_count = 0
        total_count = len(self.models)
        
        for pair_name, model_info in self.models.items():
            if self.export_model_to_onnx(pair_name, model_info):
                success_count += 1
        
        print(f"\nONNX导出完成!")
        print(f"成功导出: {success_count}/{total_count} 个模型")
        
        if success_count > 0:
            print(f"模型保存在目录: {os.path.abspath(self.onnx_models_dir)}")
            
            # 列出导出的文件
            print("\n导出的文件:")
            for file in os.listdir(self.onnx_models_dir):
                file_path = os.path.join(self.onnx_models_dir, file)
                file_size = os.path.getsize(file_path)
                print(f"  {file} ({file_size} bytes)")

def main():
    """主函数"""
    print("SVM两两情绪分类器")
    print("="*50)
    
    # 初始化SVM分类器
    classifier = SVMPairwiseClassifier(r'C:\Users\QAQ\Desktop\emotion\hrv_MK_int8.csv')
    
    # 加载数据
    classifier.load_data()
    
    # 显示选项
    print("\n请选择训练模式:")
    print("1. 训练所有SVM两两组合（推荐）")
    print("2. 训练指定的SVM两两组合")
    print("3. 自动训练所有组合并导出ONNX模型")
    
    choice = input("请输入选择 (1, 2 或 3): ").strip()
    
    if choice == "1":
        # 训练所有SVM组合
        classifier.train_all_svm_pairs()
    elif choice == "2":
        # 训练指定SVM组合
        print(f"\n可用情绪: {classifier.emotions}")
        
        emotion1 = input("请输入第一个情绪: ").strip()
        emotion2 = input("请输入第二个情绪: ").strip()
        
        if emotion1 in classifier.emotions and emotion2 in classifier.emotions and emotion1 != emotion2:
            classifier.train_svm_pair(emotion1, emotion2)
        else:
            print("输入的情绪标签无效或相同")
            return
    elif choice == "3":
        # 自动训练所有组合
        print("开始自动训练所有SVM两两组合...")
        classifier.train_all_svm_pairs()
    else:
        print("无效选择，将使用默认选项1（训练所有SVM两两组合）")
        choice = "1"
        classifier.train_all_svm_pairs()
    
    # 显示结果摘要
    classifier.show_results_summary()
    
    # 分析SVM性能
    classifier.analyze_svm_performance()
    
    # 导出所有模型为ONNX格式
    print("\n是否要导出ONNX模型？")
    export_choice = input("输入 'y' 或 'yes' 导出ONNX模型，其他键跳过: ").strip().lower()
    
    if export_choice in ['y', 'yes', '是']:
        classifier.export_all_models_to_onnx()
    else:
        print("跳过ONNX模型导出")
    
    print("\nSVM训练完成！")

if __name__ == "__main__":
    main()
