import csv
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import itertools
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

class AutoSVMPairwiseClassifier:
    def __init__(self, data_file):
        self.data_file = data_file
        self.data = []
        self.emotions = []
        self.scaler = StandardScaler()
        self.models = {}
        self.results = {}
        
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
                    float(row['SampEn'])
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

def main():
    """主函数"""
    print("自动SVM两两情绪分类器")
    print("="*50)
    
    # 初始化SVM分类器
    classifier = AutoSVMPairwiseClassifier(r'svm/test_processed.csv')
    
    # 加载数据
    classifier.load_data()
    
    # 自动训练所有SVM组合
    classifier.train_all_svm_pairs()
    
    # 显示结果摘要
    classifier.show_results_summary()
    
    # 分析SVM性能
    classifier.analyze_svm_performance()
    
    print("\nSVM训练完成！")

if __name__ == "__main__":
    main()
