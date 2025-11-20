import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix

# 读取数据
df = pd.read_csv('fortrain.csv')
print("=" * 60)
print("数据质量分析")
print("=" * 60)

print(f"\n1. 数据基本信息:")
print(f"总样本数: {len(df)}")
print(f"标签0: {(df['label']==0).sum()}")
print(f"标签1: {(df['label']==1).sum()}")

# 数据平衡
label_1_count = (df['label'] == 1).sum()
label_0_data = df[df['label'] == 0]
label_1_data = df[df['label'] == 1]
label_0_sampled = label_0_data.sample(n=label_1_count, random_state=42)
df_balanced = pd.concat([label_0_sampled, label_1_data], ignore_index=True)

print(f"\n2. 平衡后数据:")
print(f"标签0: {(df_balanced['label']==0).sum()}")
print(f"标签1: {(df_balanced['label']==1).sum()}")

# 提取特征
feature_cols = [col for col in df.columns if col not in ['label', 'subject']]
X = df_balanced[feature_cols].values
y = df_balanced['label'].values

print(f"\n3. 特征分析:")
print(f"特征维度: {X.shape[1]}")

# 检查两类数据的特征差异
label_0_features = X[y == 0]
label_1_features = X[y == 1]

print(f"\n4. 两类数据的特征差异:")
for i, col in enumerate(feature_cols[:10]):
    mean_0 = np.mean(label_0_features[:, i])
    mean_1 = np.mean(label_1_features[:, i])
    std_0 = np.std(label_0_features[:, i])
    std_1 = np.std(label_1_features[:, i])
    diff = abs(mean_1 - mean_0)
    pooled_std = np.sqrt((std_0**2 + std_1**2) / 2)
    effect_size = diff / pooled_std if pooled_std > 0 else 0
    print(f"{col:25s} | 标签0均值: {mean_0:8.4f} | 标签1均值: {mean_1:8.4f} | 差异: {diff:8.4f} | 效应量: {effect_size:.2f}")

# 标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 多次随机划分测试
print(f"\n5. 多次随机划分测试（检查是否总是100%准确率）:")
accuracies = []
for seed in range(10):
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.3, random_state=seed
    )
    
    # 使用简单参数训练
    model = SVC(kernel='rbf', C=1, gamma='scale', random_state=seed)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    accuracies.append(acc)
    print(f"随机种子 {seed:2d}: 准确率 = {acc:.4f}")

print(f"\n6. 准确率统计:")
print(f"平均准确率: {np.mean(accuracies):.4f}")
print(f"标准差: {np.std(accuracies):.4f}")
print(f"最小值: {np.min(accuracies):.4f}")
print(f"最大值: {np.max(accuracies):.4f}")

# 检查数据是否线性可分
print(f"\n7. 数据可分离性分析:")
# 计算两类数据的中心距离
center_0 = np.mean(X_scaled[y == 0], axis=0)
center_1 = np.mean(X_scaled[y == 1], axis=0)
distance = np.linalg.norm(center_1 - center_0)
print(f"两类数据中心的欧氏距离: {distance:.4f}")

# 检查是否有完全分离的特征
perfect_separation = False
for i in range(X_scaled.shape[1]):
    values_0 = X_scaled[y == 0, i]
    values_1 = X_scaled[y == 1, i]
    if np.max(values_0) < np.min(values_1) or np.max(values_1) < np.min(values_0):
        print(f"警告: 特征 {feature_cols[i]} 完全分离两类数据!")
        print(f"  标签0范围: [{np.min(values_0):.4f}, {np.max(values_0):.4f}]")
        print(f"  标签1范围: [{np.min(values_1):.4f}, {np.max(values_1):.4f}]")
        perfect_separation = True

if not perfect_separation:
    print("未发现完全分离的特征")

print(f"\n8. 结论:")
if np.mean(accuracies) > 0.99:
    print("⚠️  准确率异常高，可能的原因:")
    print("   1. 数据量太小（只有88个正样本）")
    print("   2. 特征提取过于完美地捕捉了两类差异")
    print("   3. 数据本身非常容易分类（静止vs跑步差异明显）")
    print("   4. 可能存在轻微过拟合")
    print("\n建议:")
    print("   - 增加数据量")
    print("   - 使用更严格的交叉验证")
    print("   - 检查特征工程是否过于针对训练数据")
else:
    print("准确率在合理范围内")

