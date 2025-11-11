import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 设置随机种子
def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

print("="*80)
print("使用1D-CNN对朱必艳的HRV数据进行情绪4分类")
print("="*80)

# 读取数据
df = pd.read_csv(r'1D_CNN_ZBY\zby_data.csv')
print(f"\n数据集信息:")
print(f"总样本数: {len(df)}")
print(f"特征列数: {len(df.columns) - 1}")  # 减去emotion列

# 情绪分布
print(f"\n情绪分布:")
emotion_counts = df['emotion'].value_counts()
print(emotion_counts)

# 准备数据
# 获取所有特征列（除了emotion）
feature_columns = [col for col in df.columns if col != 'emotion']
X = df[feature_columns].values
y = df['emotion'].values

print(f"\n原始数据形状: {X.shape}")
print(f"数据范围: [{np.nanmin(X):.2f}, {np.nanmax(X):.2f}]")

# 用0填充NaN值（对于长度小于最大长度的数据）
X_filled = np.nan_to_num(X, nan=0.0)
print(f"填充后数据形状: {X_filled.shape}")
print(f"填充后数据范围: [{X_filled.min():.2f}, {X_filled.max():.2f}]")

# 标签编码
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
num_classes = len(label_encoder.classes_)

print(f"\n标签编码:")
for i, emotion in enumerate(label_encoder.classes_):
    print(f"  {emotion}: {i}")

# 数据标准化（Z-score归一化，只对非零值进行）
def standardize_data(X):
    X_std = X.copy()
    # 对每个样本单独标准化（只考虑非零值）
    for i in range(X.shape[0]):
        non_zero_mask = X[i] != 0
        if non_zero_mask.sum() > 0:
            non_zero_values = X[i][non_zero_mask]
            mean = non_zero_values.mean()
            std = non_zero_values.std()
            if std > 0:
                X_std[i][non_zero_mask] = (non_zero_values - mean) / std
    return X_std

X_normalized = standardize_data(X_filled)
print(f"\n标准化后数据范围: [{X_normalized.min():.2f}, {X_normalized.max():.2f}]")

# 划分训练集和测试集 (80:20)
X_train, X_test, y_train, y_test = train_test_split(
    X_normalized, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

print(f"\n数据集划分:")
print(f"训练集: {len(X_train)} 样本")
print(f"测试集: {len(X_test)} 样本")

# 创建PyTorch数据集
class HRVDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X).unsqueeze(1)  # 添加通道维度 [N, 1, L]
        self.y = torch.LongTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_dataset = HRVDataset(X_train, y_train)
test_dataset = HRVDataset(X_test, y_test)

batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

print(f"\n数据加载器:")
print(f"批次大小: {batch_size}")
print(f"训练批次数: {len(train_loader)}")
print(f"测试批次数: {len(test_loader)}")

# 定义1D-CNN模型
class CNN1D_HRV(nn.Module):
    def __init__(self, input_length, num_classes, dropout_rate=0.5, l2_lambda=0.01):
        super(CNN1D_HRV, self).__init__()
        
        self.l2_lambda = l2_lambda
        
        # 第一层卷积
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        self.dropout1 = nn.Dropout(dropout_rate)
        
        # 第二层卷积
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(128)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(kernel_size=2)
        self.dropout2 = nn.Dropout(dropout_rate)
        
        # 第三层卷积
        self.conv3 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(256)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool1d(kernel_size=2)
        self.dropout3 = nn.Dropout(dropout_rate)
        
        # 计算卷积后的长度
        conv_output_length = input_length // 2 // 2 // 2  # 三次池化
        
        # 全连接层
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(256 * conv_output_length, 128)
        self.relu4 = nn.ReLU()
        self.dropout4 = nn.Dropout(dropout_rate)
        
        self.fc2 = nn.Linear(128, 64)
        self.relu5 = nn.ReLU()
        self.dropout5 = nn.Dropout(dropout_rate)
        
        self.fc3 = nn.Linear(64, num_classes)
    
    def forward(self, x):
        # 卷积块1
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.dropout1(x)
        
        # 卷积块2
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = self.dropout2(x)
        
        # 卷积块3
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.pool3(x)
        x = self.dropout3(x)
        
        # 全连接层
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu4(x)
        x = self.dropout4(x)
        
        x = self.fc2(x)
        x = self.relu5(x)
        x = self.dropout5(x)
        
        x = self.fc3(x)
        
        return x
    
    def get_l2_loss(self):
        """计算L2正则化损失"""
        l2_loss = 0
        for param in self.parameters():
            l2_loss += torch.sum(param ** 2)
        return self.l2_lambda * l2_loss

# 创建模型
input_length = X_train.shape[1]
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\n使用设备: {device}")

model = CNN1D_HRV(
    input_length=input_length,
    num_classes=num_classes,
    dropout_rate=0.5,
    l2_lambda=0.001
).to(device)

print(f"\n模型架构:")
print(model)

# 计算模型参数量
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\n总参数量: {total_params:,}")
print(f"可训练参数量: {trainable_params:,}")

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)  # 额外的L2正则化
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)

# 训练函数
def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        
        # 交叉熵损失
        ce_loss = criterion(outputs, labels)
        
        # L2正则化损失
        l2_loss = model.get_l2_loss()
        
        # 总损失
        loss = ce_loss + l2_loss
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100 * correct / total
    
    return epoch_loss, epoch_acc

# 验证函数
def validate_epoch(model, test_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    epoch_loss = running_loss / len(test_loader)
    epoch_acc = 100 * correct / total
    
    return epoch_loss, epoch_acc, all_preds, all_labels

# 训练模型
num_epochs = 100
print(f"\n{'='*80}")
print(f"开始训练 (共 {num_epochs} 轮)")
print(f"{'='*80}\n")

train_losses = []
train_accs = []
test_losses = []
test_accs = []
best_test_acc = 0
best_epoch = 0

for epoch in range(num_epochs):
    train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
    test_loss, test_acc, _, _ = validate_epoch(model, test_loader, criterion, device)
    
    train_losses.append(train_loss)
    train_accs.append(train_acc)
    test_losses.append(test_loss)
    test_accs.append(test_acc)
    
    # 学习率调度
    scheduler.step(test_loss)
    
    # 保存最佳模型
    if test_acc > best_test_acc:
        best_test_acc = test_acc
        best_epoch = epoch
        torch.save(model.state_dict(), '朱必艳_1dcnn_best_model.pth')
    
    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}]")
        print(f"  训练 - Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%")
        print(f"  测试 - Loss: {test_loss:.4f}, Acc: {test_acc:.2f}%")

print(f"\n{'='*80}")
print(f"训练完成!")
print(f"最佳测试准确率: {best_test_acc:.2f}% (Epoch {best_epoch+1})")
print(f"{'='*80}\n")

# 加载最佳模型
model.load_state_dict(torch.load('朱必艳_1dcnn_best_model.pth'))

# 最终评估
_, final_test_acc, final_preds, final_labels = validate_epoch(model, test_loader, criterion, device)

print(f"\n最终测试集评估:")
print(f"准确率: {final_test_acc:.2f}%")

# 分类报告
print(f"\n详细分类报告:")
print(classification_report(
    final_labels, 
    final_preds, 
    target_names=label_encoder.classes_,
    digits=4
))

# 混淆矩阵
cm = confusion_matrix(final_labels, final_preds)
print(f"\n混淆矩阵:")
print(cm)

# 可视化
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 1. 训练和测试损失
ax1 = axes[0, 0]
ax1.plot(train_losses, label='训练损失', linewidth=2)
ax1.plot(test_losses, label='测试损失', linewidth=2)
ax1.set_xlabel('Epoch', fontsize=12)
ax1.set_ylabel('Loss', fontsize=12)
ax1.set_title('训练和测试损失曲线', fontsize=14, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 2. 训练和测试准确率
ax2 = axes[0, 1]
ax2.plot(train_accs, label='训练准确率', linewidth=2)
ax2.plot(test_accs, label='测试准确率', linewidth=2)
ax2.axhline(y=best_test_acc, color='r', linestyle='--', label=f'最佳测试准确率: {best_test_acc:.2f}%')
ax2.set_xlabel('Epoch', fontsize=12)
ax2.set_ylabel('Accuracy (%)', fontsize=12)
ax2.set_title('训练和测试准确率曲线', fontsize=14, fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)

# 3. 混淆矩阵热图
ax3 = axes[1, 0]
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=label_encoder.classes_,
            yticklabels=label_encoder.classes_,
            ax=ax3, cbar_kws={'label': '样本数'})
ax3.set_xlabel('预测标签', fontsize=12)
ax3.set_ylabel('真实标签', fontsize=12)
ax3.set_title('混淆矩阵', fontsize=14, fontweight='bold')

# 4. 各类别准确率
ax4 = axes[1, 1]
class_accuracies = []
for i in range(len(label_encoder.classes_)):
    class_mask = np.array(final_labels) == i
    if class_mask.sum() > 0:
        class_acc = 100 * np.sum((np.array(final_preds)[class_mask] == i)) / class_mask.sum()
        class_accuracies.append(class_acc)
    else:
        class_accuracies.append(0)

bars = ax4.bar(label_encoder.classes_, class_accuracies, color=['skyblue', 'lightcoral', 'lightgreen', 'lightyellow'])
ax4.set_xlabel('情绪类别', fontsize=12)
ax4.set_ylabel('准确率 (%)', fontsize=12)
ax4.set_title('各情绪类别分类准确率', fontsize=14, fontweight='bold')
ax4.set_ylim(0, 100)
ax4.grid(True, alpha=0.3, axis='y')

for bar, acc in zip(bars, class_accuracies):
    height = bar.get_height()
    ax4.text(bar.get_x() + bar.get_width()/2., height,
            f'{acc:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig('朱必艳_1dcnn_training_results.png', dpi=300, bbox_inches='tight')
print(f"\n[OK] 训练结果可视化已保存到: 朱必艳_1dcnn_training_results.png")
plt.show()

# 保存训练历史
history_df = pd.DataFrame({
    'epoch': range(1, num_epochs + 1),
    'train_loss': train_losses,
    'train_acc': train_accs,
    'test_loss': test_losses,
    'test_acc': test_accs
})
history_df.to_csv('朱必艳_1dcnn_training_history.csv', index=False, encoding='utf-8-sig')
print(f"[OK] 训练历史已保存到: 朱必艳_1dcnn_training_history.csv")

print(f"\n[OK] 最佳模型已保存到: 朱必艳_1dcnn_best_model.pth")
