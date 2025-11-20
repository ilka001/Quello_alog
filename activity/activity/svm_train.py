import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix

# ONNX导出相关
try:
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType
    import onnxruntime as ort
    HAS_ONNX = True
except ImportError:
    HAS_ONNX = False
    print("警告: skl2onnx或onnxruntime未安装，将跳过ONNX导出功能")
    print("安装命令: pip install skl2onnx onnxruntime")
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_PLOTTING = True
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
    plt.rcParams['axes.unicode_minus'] = False
except ImportError:
    HAS_PLOTTING = False
    print("警告: matplotlib/seaborn未安装，将跳过可视化部分")

import warnings
warnings.filterwarnings('ignore')

def export_to_onnx(model, scaler, feature_names, output_file='svm_model.onnx'):
    """将SVM模型和标准化器导出为ONNX格式
    
    参数:
        model: 训练好的SVM模型
        scaler: 标准化器
        feature_names: 特征名称列表
        output_file: 输出ONNX文件路径
    """
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType
    from sklearn.pipeline import Pipeline
    import onnxruntime as ort
    import numpy as np
    pipeline = Pipeline([
        ('scaler', scaler),
        ('svm', model)
    ])
    
    # 定义输入类型（18维特征）
    initial_type = [('float_input', FloatTensorType([None, len(feature_names)]))]
    
    # 转换为ONNX
    onnx_model = convert_sklearn(
        pipeline,
        initial_types=initial_type,
        target_opset=12  # 使用较新的opset版本
    )
    
    # 保存ONNX模型
    with open(output_file, 'wb') as f:
        f.write(onnx_model.SerializeToString())
    
    print(f"  ONNX模型已保存到: {output_file}")
    print(f"  输入维度: {len(feature_names)}维特征")
    print(f"  输出: 类别标签 (0或1)")
    
    # 验证ONNX模型
    try:
        session = ort.InferenceSession(output_file)
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name
        
        # 使用一个示例输入测试
        test_input = np.array([[0.0] * len(feature_names)], dtype=np.float32)
        result = session.run([output_name], {input_name: test_input})
        
        print(f"  ONNX模型验证成功")
        print(f"  输入名称: {input_name}")
        print(f"  输出名称: {output_name}")
        print(f"  测试预测结果: {result[0]}")
    except Exception as e:
        print(f"  ONNX模型验证警告: {e}")
    
    return output_file

def balance_data(df, random_state=None):
    """平衡数据：选取和标签1等量的标签0数据（随机选取）
    如果random_state为None，则每次运行都是随机的
    """
    label_1_count = (df['label'] == 1).sum()
    label_0_data = df[df['label'] == 0]
    label_1_data = df[df['label'] == 1]
    
    # 随机采样标签0的数据，使其数量等于标签1（每次随机）
    label_0_sampled = label_0_data.sample(n=label_1_count, random_state=random_state)
    
    # 合并数据并打乱
    balanced_df = pd.concat([label_0_sampled, label_1_data], ignore_index=True)
    balanced_df = balanced_df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    
    print(f"原始数据: 标签0={len(label_0_data)}, 标签1={len(label_1_data)}")
    print(f"平衡后数据: 标签0={len(label_0_sampled)}, 标签1={len(label_1_data)}")
    
    return balanced_df

def split_by_sample(df, test_size=0.3, random_state=None):
    """按样本总量划分训练集和测试集（7:3），保证两个标签平衡
    如果random_state为None，则每次运行都是随机的
    """
    label_0_data = df[df['label'] == 0].copy()
    label_1_data = df[df['label'] == 1].copy()
    
    # 分别对两个标签的数据进行划分
    train_0, test_0 = train_test_split(
        label_0_data, test_size=test_size, random_state=random_state
    )
    train_1, test_1 = train_test_split(
        label_1_data, test_size=test_size, random_state=random_state
    )
    
    # 合并训练集和测试集
    train_df = pd.concat([train_0, train_1], ignore_index=True)
    test_df = pd.concat([test_0, test_1], ignore_index=True)
    
    # 打乱顺序（保持随机性）
    train_df = train_df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    test_df = test_df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    
    print(f"\n按样本划分（7:3）:")
    print(f"训练集样本数: {len(train_df)}")
    print(f"测试集样本数: {len(test_df)}")
    print(f"训练集标签分布: 标签0={len(train_0)}, 标签1={len(train_1)}")
    print(f"测试集标签分布: 标签0={len(test_0)}, 标签1={len(test_1)}")
    print(f"训练集标签0占比: {len(train_0)/len(train_df):.2%}, 标签1占比: {len(train_1)/len(train_df):.2%}")
    print(f"测试集标签0占比: {len(test_0)/len(test_df):.2%}, 标签1占比: {len(test_1)/len(test_df):.2%}")
    
    return train_df, test_df

def main():
    import time
    
    print("=" * 60)
    print("SVM模型训练")
    print("=" * 60)
    
    # 生成随机种子（使用时间戳，每次运行都不同）
    random_seed = int(time.time()) % 10000
    print(f"\n本次运行使用的随机种子: {random_seed}")
    print("注意: 每次运行都会使用不同的随机种子，结果会有所不同\n")
    
    # 1. 读取数据
    print("1. 读取数据...")
    df = pd.read_csv('fortrain.csv')
    print(f"原始数据形状: {df.shape}")
    
    # 2. 数据平衡（使用随机种子，每次运行都不同）
    print("\n2. 数据平衡...")
    df_balanced = balance_data(df, random_state=random_seed)
    
    # 3. 按样本划分训练集和测试集（7:3），保证标签平衡
    print("\n3. 按样本划分数据集（7:3）...")
    train_df, test_df = split_by_sample(df_balanced, test_size=0.3, random_state=random_seed)
    
    # 4. 提取特征和标签
    feature_cols = [col for col in df.columns if col not in ['label', 'subject']]
    X_train = train_df[feature_cols].values
    y_train = train_df['label'].values
    X_test = test_df[feature_cols].values
    y_test = test_df['label'].values
    
    print(f"\n特征维度: {len(feature_cols)}")
    print(f"训练集: {X_train.shape}, 测试集: {X_test.shape}")
    
    # 5. Z-score标准化
    print("\n4. Z-score标准化...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print("标准化完成: 均值≈0, 方差≈1")
    print(f"训练集标准化后均值: {np.mean(X_train_scaled, axis=0)[:3]}")
    print(f"训练集标准化后方差: {np.var(X_train_scaled, axis=0)[:3]}")
    
    # 6. 网格搜索和交叉验证进行超参数调优
    print("\n5. 网格搜索和交叉验证进行超参数调优...")
    param_grid = {
        'C': [0.1, 1, 10, 100],
        'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1],
        'kernel': ['rbf', 'linear', 'poly']
    }
    
    # 使用随机种子初始化SVM
    svm = SVC(random_state=random_seed)
    # GridSearchCV本身不支持random_state参数，但可以通过cv参数传递随机种子
    from sklearn.model_selection import StratifiedKFold
    cv_fold = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_seed)
    grid_search = GridSearchCV(
        svm, param_grid, cv=cv_fold, scoring='f1', n_jobs=-1, verbose=1
    )
    
    print("开始网格搜索...")
    grid_search.fit(X_train_scaled, y_train)
    
    print(f"\n最佳参数: {grid_search.best_params_}")
    print(f"最佳交叉验证F1分数: {grid_search.best_score_:.4f}")
    
    # 7. 使用最佳模型进行预测
    print("\n6. 使用最佳模型进行预测...")
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test_scaled)
    
    # 8. 评估指标
    print("\n7. 模型评估...")
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    print(f"\n评估指标:")
    print(f"准确率 (Accuracy): {accuracy:.4f}")
    print(f"精确率 (Precision): {precision:.4f}")
    print(f"召回率 (Recall): {recall:.4f}")
    print(f"F1-score: {f1:.4f}")
    
    print("\n分类报告:")
    print(classification_report(y_test, y_pred, target_names=['静止(0)', '跑步(1)']))
    
    # 9. 导出ONNX模型
    if HAS_ONNX:
        print("\n9. 导出ONNX模型...")
        try:
            export_to_onnx(best_model, scaler, feature_cols, output_file='svm_model.onnx')
            print("ONNX模型导出成功！")
        except Exception as e:
            print(f"ONNX模型导出失败: {e}")
    else:
        print("\n9. ONNX导出跳过（库未安装）")
    
    # 10. 可视化
    if HAS_PLOTTING:
        print("\n10. 可视化结果...")
        
        # 创建图形
        fig = plt.figure(figsize=(16, 10))
        
        # 9.1 评估指标柱状图
        ax1 = plt.subplot(2, 3, 1)
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-score']
        values = [accuracy, precision, recall, f1]
        colors = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12']
        bars = ax1.bar(metrics, values, color=colors, alpha=0.7, edgecolor='black')
        ax1.set_ylim([0, 1])
        ax1.set_ylabel('分数', fontsize=12)
        ax1.set_title('模型评估指标', fontsize=14, fontweight='bold')
        ax1.grid(axis='y', alpha=0.3)
        
        # 在柱状图上添加数值标签
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{val:.4f}', ha='center', va='bottom', fontsize=10)
        
        # 9.2 混淆矩阵
        ax2 = plt.subplot(2, 3, 2)
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax2,
                    xticklabels=['静止(0)', '跑步(1)'],
                    yticklabels=['静止(0)', '跑步(1)'])
        ax2.set_xlabel('预测标签', fontsize=12)
        ax2.set_ylabel('真实标签', fontsize=12)
        ax2.set_title('混淆矩阵', fontsize=14, fontweight='bold')
        
        # 9.3 交叉验证分数分布
        ax3 = plt.subplot(2, 3, 3)
        cv_scores = cross_val_score(best_model, X_train_scaled, y_train, cv=5, scoring='f1')
        ax3.boxplot([cv_scores], labels=['F1-score'])
        ax3.scatter([1] * len(cv_scores), cv_scores, alpha=0.5, color='red', s=50)
        ax3.set_ylabel('F1-score', fontsize=12)
        ax3.set_title('5折交叉验证分数分布', fontsize=14, fontweight='bold')
        ax3.grid(axis='y', alpha=0.3)
        
        # 9.4 特征重要性
        ax4 = plt.subplot(2, 3, 4)
        try:
            if grid_search.best_params_['kernel'] == 'linear':
                feature_importance = np.abs(best_model.coef_[0])
                print("  使用线性核的系数计算特征重要性")
            else:
                # 对于非线性核，使用permutation importance
                print("  计算特征重要性（permutation importance）...")
                from sklearn.inspection import permutation_importance
                perm_importance = permutation_importance(
                    best_model, X_test_scaled, y_test, n_repeats=5, random_state=random_seed, n_jobs=-1, scoring='f1'
                )
                feature_importance = perm_importance.importances_mean
                print(f"  特征重要性计算完成，形状: {feature_importance.shape}")
                print(f"  特征重要性范围: [{np.min(feature_importance):.6f}, {np.max(feature_importance):.6f}]")
                
                # 如果所有值都很小或为0，尝试使用方差作为替代
                if np.max(feature_importance) < 1e-6 or np.all(feature_importance == 0):
                    print("  特征重要性值过小，使用特征方差作为替代")
                    feature_importance = np.var(X_train_scaled, axis=0)
            
            # 检查特征重要性
            if len(feature_importance) == 0:
                ax4.text(0.5, 0.5, '特征重要性为空', 
                        ha='center', va='center', transform=ax4.transAxes, fontsize=12)
                ax4.set_title('Top 10 特征重要性', fontsize=14, fontweight='bold')
            elif np.all(np.isnan(feature_importance)) or np.all(feature_importance == 0):
                ax4.text(0.5, 0.5, '所有特征重要性为0\n或包含NaN值', 
                        ha='center', va='center', transform=ax4.transAxes, fontsize=12)
                ax4.set_title('Top 10 特征重要性', fontsize=14, fontweight='bold')
            else:
                # 处理NaN值（如果有）
                feature_importance = np.nan_to_num(feature_importance, nan=0.0)
                
                # 选择Top N个特征（最多10个，或所有特征如果少于10个）
                n_features = min(10, len(feature_importance))
                top_features_idx = np.argsort(feature_importance)[-n_features:][::-1]
                top_features_names = [feature_cols[i] for i in top_features_idx]
                top_features_values = feature_importance[top_features_idx]
                
                # 确保至少有一些值要显示
                if len(top_features_names) > 0 and np.any(top_features_values > 0):
                    bars = ax4.barh(range(len(top_features_names)), top_features_values, color='#9b59b6', alpha=0.7)
                    ax4.set_yticks(range(len(top_features_names)))
                    ax4.set_yticklabels(top_features_names, fontsize=9)
                    ax4.set_xlabel('重要性', fontsize=12)
                    ax4.set_title(f'Top {len(top_features_names)} 特征重要性', fontsize=14, fontweight='bold')
                    ax4.grid(axis='x', alpha=0.3)
                    
                    # 如果值太小，调整x轴范围使其可见
                    if np.max(top_features_values) < 0.01:
                        ax4.set_xlim(0, max(0.01, np.max(top_features_values) * 1.1))
                    
                    print(f"  显示Top {len(top_features_names)}个特征，最大值: {np.max(top_features_values):.6f}")
                else:
                    ax4.text(0.5, 0.5, f'特征重要性值过小\n最大值: {np.max(feature_importance):.6f}', 
                            ha='center', va='center', transform=ax4.transAxes, fontsize=10)
                    ax4.set_title('Top 10 特征重要性', fontsize=14, fontweight='bold')
        except Exception as e:
            import traceback
            print(f"  特征重要性计算出错: {e}")
            print(f"  错误详情: {traceback.format_exc()}")
            ax4.text(0.5, 0.5, f'特征重要性计算出错\n{str(e)[:50]}...', 
                    ha='center', va='center', transform=ax4.transAxes, fontsize=10)
            ax4.set_title('Top 10 特征重要性', fontsize=14, fontweight='bold')
        
        # 9.5 训练集和测试集标签分布
        ax5 = plt.subplot(2, 3, 5)
        train_labels = [np.sum(y_train == 0), np.sum(y_train == 1)]
        test_labels = [np.sum(y_test == 0), np.sum(y_test == 1)]
        x = np.arange(2)
        width = 0.35
        ax5.bar(x - width/2, train_labels, width, label='训练集', color='#3498db', alpha=0.7)
        ax5.bar(x + width/2, test_labels, width, label='测试集', color='#e74c3c', alpha=0.7)
        ax5.set_xlabel('标签', fontsize=12)
        ax5.set_ylabel('样本数', fontsize=12)
        ax5.set_title('数据集标签分布', fontsize=14, fontweight='bold')
        ax5.set_xticks(x)
        ax5.set_xticklabels(['静止(0)', '跑步(1)'])
        ax5.legend()
        ax5.grid(axis='y', alpha=0.3)
        
        # 9.6 超参数搜索热力图（C vs gamma，使用rbf核）
        ax6 = plt.subplot(2, 3, 6)
        if grid_search.best_params_['kernel'] == 'rbf':
            # 提取rbf核的结果
            rbf_results = []
            for params, score in zip(grid_search.cv_results_['params'], grid_search.cv_results_['mean_test_score']):
                if params['kernel'] == 'rbf':
                    rbf_results.append((params['C'], params['gamma'], score))
            
            if rbf_results:
                C_values = sorted(set([r[0] for r in rbf_results]))
                gamma_values = sorted(set([r[1] for r in rbf_results]), key=lambda x: str(x))
                
                # 创建热力图数据
                heatmap_data = np.zeros((len(gamma_values), len(C_values)))
                for c, g, s in rbf_results:
                    c_idx = C_values.index(c)
                    g_idx = gamma_values.index(g)
                    heatmap_data[g_idx, c_idx] = s
                
                sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='YlOrRd', ax=ax6,
                           xticklabels=[f'C={c}' for c in C_values],
                           yticklabels=[f'γ={g}' for g in gamma_values])
                ax6.set_xlabel('C (正则化参数)', fontsize=12)
                ax6.set_ylabel('γ (RBF核参数)', fontsize=12)
                ax6.set_title('超参数搜索热力图 (RBF核)', fontsize=14, fontweight='bold')
            else:
                ax6.text(0.5, 0.5, '最佳核函数不是RBF\n无法显示热力图', 
                        ha='center', va='center', transform=ax6.transAxes, fontsize=12)
                ax6.set_title('超参数搜索', fontsize=14, fontweight='bold')
        else:
            ax6.text(0.5, 0.5, f"最佳核函数: {grid_search.best_params_['kernel']}\n\n最佳参数:\nC={grid_search.best_params_['C']}\nγ={grid_search.best_params_['gamma']}", 
                    ha='center', va='center', transform=ax6.transAxes, fontsize=12)
            ax6.set_title('最佳超参数', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.show()
        
        print("\n可视化完成！")
    else:
        print("\n8. 可视化跳过（matplotlib未安装）")

if __name__ == '__main__':
    main()

