import csv
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# 设置随机种子
torch.manual_seed(42)
np.random.seed(42)

# 数据加载
class EmotionDataset(Dataset):
    def __init__(self, csv_file):
        """
        加载CSV文件，每行是一个样本
        """
        self.data = []
        self.labels = []
        
        with open(csv_file, 'r', encoding='utf-8-sig') as f:
            reader = csv.reader(f)
            for row in reader:
                if row:
                    # 最后一列是标签
                    label = row[-1]
                    # 其余列是特征值，转换为浮点数
                    features = [float(x) for x in row[:-1]]
                    
                    self.data.append(features)
                    self.labels.append(label)
        
        # 将标签编码为数字
        self.label_encoder = LabelEncoder()
        self.labels = self.label_encoder.fit_transform(self.labels)
        
        # 找到最大长度并进行padding
        self.max_length = max(len(x) for x in self.data)
        
        # 将数据padding到相同长度
        self.data_padded = []
        for features in self.data:
            if len(features) < self.max_length:
                # 用0填充
                padded = features + [0] * (self.max_length - len(features))
            else:
                padded = features
            self.data_padded.append(padded)
        
        # 转换为numpy数组
        self.data_padded = np.array(self.data_padded, dtype=np.float32)
        self.labels = np.array(self.labels, dtype=np.int64)
        
        # 数据标准化
        self.mean = np.mean(self.data_padded[self.data_padded != 0])
        self.std = np.std(self.data_padded[self.data_padded != 0])
        
        # 只标准化非零值
        mask = self.data_padded != 0
        self.data_padded[mask] = (self.data_padded[mask] - self.mean) / self.std
        
        print(f"数据集信息:")
        print(f"  样本数: {len(self.data)}")
        print(f"  序列最大长度: {self.max_length}")
        print(f"  类别数: {len(self.label_encoder.classes_)}")
        print(f"  类别标签: {self.label_encoder.classes_}")
        print(f"  各类别样本数: {np.bincount(self.labels)}")
    
    def __len__(self):
        return len(self.data_padded)
    
    def __getitem__(self, idx):
        # 返回形状: (1, sequence_length) - 单通道时间序列
        x = torch.FloatTensor(self.data_padded[idx]).unsqueeze(0)
        y = torch.LongTensor([self.labels[idx]])[0]
        return x, y


# 1D-CNN模型（增强正则化版本）
class CNN1D(nn.Module):
    def __init__(self, input_length, num_classes):
        super(CNN1D, self).__init__()
        
        # 第一个卷积块
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(16)
        self.dropout_conv1 = nn.Dropout(0.2)
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        
        # 第二个卷积块
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(32)
        self.dropout_conv2 = nn.Dropout(0.3)
        self.pool2 = nn.MaxPool1d(kernel_size=2)
        
        # 第三个卷积块
        self.conv3 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(64)
        self.dropout_conv3 = nn.Dropout(0.4)
        self.pool3 = nn.MaxPool1d(kernel_size=2)
        
        # 计算全连接层的输入大小
        self.flatten_size = 64 * (input_length // 8)
        
        # 全连接层
        self.fc1 = nn.Linear(self.flatten_size, 128)
        self.dropout1 = nn.Dropout(0.6)
        self.fc2 = nn.Linear(128, 64)
        self.dropout2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(64, num_classes)
        
        # 激活函数
        self.relu = nn.ReLU()
        
    def forward(self, x):
        # 卷积块1
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout_conv1(x)
        x = self.pool1(x)
        
        # 卷积块2
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.dropout_conv2(x)
        x = self.pool2(x)
        
        # 卷积块3
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.dropout_conv3(x)
        x = self.pool3(x)
        
        # 展平
        x = x.view(x.size(0), -1)
        
        # 全连接层
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout1(x)
        
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout2(x)
        
        x = self.fc3(x)
        
        return x


# 训练函数
def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100 * correct / total
    return epoch_loss, epoch_acc


# 评估函数
def evaluate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100 * correct / total
    return epoch_loss, epoch_acc


# 主程序
if __name__ == '__main__':
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}\n")
    
    # 加载数据
    data_file = r'1D_CNN_LJY\ljy_consolidated.csv'
    print("加载数据...")
    dataset = EmotionDataset(data_file)
    
    # 划分训练集和测试集 (80% 训练, 20% 测试)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    
    print(f"\n训练集大小: {train_size}")
    print(f"测试集大小: {test_size}")
    
    # 创建数据加载器
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # 创建模型
    num_classes = len(dataset.label_encoder.classes_)
    model = CNN1D(input_length=dataset.max_length, num_classes=num_classes)
    model = model.to(device)
    
    print(f"\n模型结构:")
    print(model)
    print(f"\n总参数数量: {sum(p.numel() for p in model.parameters())}")
    
    # 定义损失函数和优化器（添加L2正则化）
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-4)  # 降低学习率，添加L2正则化
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=10, factor=0.5)  # 增加耐心值
    
    # 训练
    num_epochs = 200  # 训练200轮
    train_losses = []
    train_accs = []
    test_losses = []
    test_accs = []
    
    best_test_acc = 0.0
    
    print("\n开始训练...\n")
    print("正则化策略:")
    print("  - L2正则化 (weight_decay=1e-4)")
    print("  - 增强Dropout (卷积层: 0.2-0.4, 全连接层: 0.5-0.6)")
    print("  - 减少模型参数量")
    print("  - 训练轮数: 200 (无早停)")
    print("  - 初始学习率: 0.0005")
    print()
    print(f"{'Epoch':>5} {'Train Loss':>12} {'Train Acc':>12} {'Test Loss':>12} {'Test Acc':>12}")
    print("-" * 65)
    
    for epoch in range(num_epochs):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        
        # 学习率调度
        scheduler.step(test_loss)
        
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        test_losses.append(test_loss)
        test_accs.append(test_acc)
        
        # 保存最佳模型
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            torch.save(model.state_dict(), '李佳宜_best_model_regularized.pth')
        
        print(f"{epoch+1:>5} {train_loss:>12.4f} {train_acc:>11.2f}% {test_loss:>12.4f} {test_acc:>11.2f}%")
    
    print("\n" + "="*65)
    print(f"训练完成！最佳测试准确率: {best_test_acc:.2f}%")
    print(f"模型已保存为: 李佳宜_best_model_regularized.pth")
    
    # 加载最佳模型进行测试集评估
    print("\n加载最佳模型进行详细评估...")
    model.load_state_dict(torch.load('李佳宜_best_model_regularized.pth'))
    model.eval()
    
    # 收集测试集的预测结果和真实标签
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # 计算混淆矩阵
    cm = confusion_matrix(all_labels, all_preds)
    
    # 分类报告
    print("\n" + "="*65)
    print("测试集分类报告:")
    print("="*65)
    class_names = dataset.label_encoder.classes_
    print(classification_report(all_labels, all_preds, target_names=class_names, digits=4))
    
    # 绘制训练曲线和混淆矩阵
    fig = plt.figure(figsize=(18, 5))
    
    # 子图1: Loss曲线
    plt.subplot(1, 3, 1)
    plt.plot(train_losses, label='Train Loss', linewidth=2)
    plt.plot(test_losses, label='Test Loss', linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Loss Curve (Regularized)', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # 子图2: Accuracy曲线
    plt.subplot(1, 3, 2)
    plt.plot(train_accs, label='Train Accuracy', linewidth=2)
    plt.plot(test_accs, label='Test Accuracy', linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.title('Accuracy Curve (Regularized)', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # 子图3: 混淆矩阵
    plt.subplot(1, 3, 3)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Count'})
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.title('Confusion Matrix on Test Set', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('李佳宜_training_results.png', dpi=300, bbox_inches='tight')
    print("\n综合结果图已保存为: 李佳宜_training_results.png")
    
    # 单独绘制更大的混淆矩阵
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='YlGnBu', 
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Sample Count'},
                annot_kws={'size': 14})
    plt.xlabel('Predicted Label', fontsize=14, fontweight='bold')
    plt.ylabel('True Label', fontsize=14, fontweight='bold')
    plt.title('Confusion Matrix - Test Set (李佳宜)', fontsize=16, fontweight='bold')
    
    # 计算每个类别的准确率并添加到图中
    for i, class_name in enumerate(class_names):
        class_acc = cm[i, i] / cm[i].sum() * 100
        plt.text(len(class_names) + 0.5, i + 0.5, f'{class_acc:.1f}%', 
                ha='center', va='center', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('李佳宜_confusion_matrix.png', dpi=300, bbox_inches='tight')
    print("详细混淆矩阵已保存为: 李佳宜_confusion_matrix.png")
    
    print("\n类别映射:")
    for i, label in enumerate(dataset.label_encoder.classes_):
        print(f"  {i}: {label}")
