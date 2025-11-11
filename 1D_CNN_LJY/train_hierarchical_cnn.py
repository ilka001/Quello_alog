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
        
        # 原始标签
        self.original_labels = np.array(self.labels.copy())
        
        # 创建两级标签
        # 第一级：分为两大组
        # 组0: 平静、悲伤 (低唤醒度)
        # 组1: 愉悦、焦虑 (高唤醒度)
        self.group_labels = []
        for label in self.labels:
            if label in ['平静', '悲伤']:
                self.group_labels.append(0)  # 低唤醒度组
            else:  # 愉悦、焦虑
                self.group_labels.append(1)  # 高唤醒度组
        
        self.group_labels = np.array(self.group_labels, dtype=np.int64)
        
        # 第二级：组内细分
        # 组0内：0-平静，1-悲伤
        # 组1内：0-愉悦，1-焦虑
        self.subgroup_labels = []
        for label in self.labels:
            if label == '平静':
                self.subgroup_labels.append(0)
            elif label == '悲伤':
                self.subgroup_labels.append(1)
            elif label == '愉悦':
                self.subgroup_labels.append(0)
            elif label == '焦虑':
                self.subgroup_labels.append(1)
        
        self.subgroup_labels = np.array(self.subgroup_labels, dtype=np.int64)
        
        # 找到最大长度并进行padding
        self.max_length = max(len(x) for x in self.data)
        
        # 将数据padding到相同长度
        self.data_padded = []
        for features in self.data:
            if len(features) < self.max_length:
                padded = features + [0] * (self.max_length - len(features))
            else:
                padded = features
            self.data_padded.append(padded)
        
        # 转换为numpy数组
        self.data_padded = np.array(self.data_padded, dtype=np.float32)
        
        # 数据标准化
        self.mean = np.mean(self.data_padded[self.data_padded != 0])
        self.std = np.std(self.data_padded[self.data_padded != 0])
        
        # 只标准化非零值
        mask = self.data_padded != 0
        self.data_padded[mask] = (self.data_padded[mask] - self.mean) / self.std
        
        print(f"数据集信息:")
        print(f"  样本数: {len(self.data)}")
        print(f"  序列最大长度: {self.max_length}")
        print(f"  第一级分类 - 两大组:")
        print(f"    组0(低唤醒): {np.sum(self.group_labels == 0)} 个 (平静+悲伤)")
        print(f"    组1(高唤醒): {np.sum(self.group_labels == 1)} 个 (愉悦+焦虑)")
        print(f"  第二级分类 - 组内细分:")
        
        # 统计每个原始类别的数量
        unique, counts = np.unique(self.original_labels, return_counts=True)
        for label, count in zip(unique, counts):
            print(f"    {label}: {count} 个")
    
    def __len__(self):
        return len(self.data_padded)
    
    def __getitem__(self, idx):
        x = torch.FloatTensor(self.data_padded[idx]).unsqueeze(0)
        group_label = torch.LongTensor([self.group_labels[idx]])[0]
        subgroup_label = torch.LongTensor([self.subgroup_labels[idx]])[0]
        original_label = self.original_labels[idx]
        return x, group_label, subgroup_label, original_label


# 1D-CNN模型（用于两级分类）
class CNN1D_Classifier(nn.Module):
    def __init__(self, input_length, num_classes):
        super(CNN1D_Classifier, self).__init__()
        
        # 卷积块
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(16)
        self.dropout_conv1 = nn.Dropout(0.2)
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(32)
        self.dropout_conv2 = nn.Dropout(0.3)
        self.pool2 = nn.MaxPool1d(kernel_size=2)
        
        self.conv3 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(64)
        self.dropout_conv3 = nn.Dropout(0.4)
        self.pool3 = nn.MaxPool1d(kernel_size=2)
        
        # 计算全连接层的输入大小
        self.flatten_size = 64 * (input_length // 8)
        
        # 全连接层
        self.fc1 = nn.Linear(self.flatten_size, 128)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 64)
        self.dropout2 = nn.Dropout(0.4)
        self.fc3 = nn.Linear(64, num_classes)
        
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout_conv1(x)
        x = self.pool1(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.dropout_conv2(x)
        x = self.pool2(x)
        
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.dropout_conv3(x)
        x = self.pool3(x)
        
        x = x.view(x.size(0), -1)
        
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout1(x)
        
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout2(x)
        
        x = self.fc3(x)
        
        return x


# 双层分类系统
class HierarchicalClassifier:
    def __init__(self, input_length, device):
        self.device = device
        
        # 第一级：分组模型 (2分类：低唤醒 vs 高唤醒)
        self.group_model = CNN1D_Classifier(input_length, num_classes=2).to(device)
        
        # 第二级：组内细分模型
        # 低唤醒组内分类器 (2分类：平静 vs 悲伤)
        self.low_arousal_model = CNN1D_Classifier(input_length, num_classes=2).to(device)
        
        # 高唤醒组内分类器 (2分类：愉悦 vs 焦虑)
        self.high_arousal_model = CNN1D_Classifier(input_length, num_classes=2).to(device)
        
        print("\n双层分类系统结构:")
        print("="*60)
        print("第一级模型: 分组分类器")
        print(f"  - 输入: 时间序列数据")
        print(f"  - 输出: 2类 (低唤醒组 vs 高唤醒组)")
        print(f"  - 参数量: {sum(p.numel() for p in self.group_model.parameters())}")
        print()
        print("第二级模型A: 低唤醒组内分类器")
        print(f"  - 输入: 被第一级分为低唤醒的样本")
        print(f"  - 输出: 2类 (平静 vs 悲伤)")
        print(f"  - 参数量: {sum(p.numel() for p in self.low_arousal_model.parameters())}")
        print()
        print("第二级模型B: 高唤醒组内分类器")
        print(f"  - 输入: 被第一级分为高唤醒的样本")
        print(f"  - 输出: 2类 (愉悦 vs 焦虑)")
        print(f"  - 参数量: {sum(p.numel() for p in self.high_arousal_model.parameters())}")
        print("="*60)
    
    def predict(self, x):
        """
        双层预测
        返回: (group_pred, final_pred)
        final_pred: 0-平静, 1-悲伤, 2-愉悦, 3-焦虑
        """
        self.group_model.eval()
        self.low_arousal_model.eval()
        self.high_arousal_model.eval()
        
        with torch.no_grad():
            # 第一级：判断是低唤醒还是高唤醒
            group_output = self.group_model(x)
            group_pred = torch.argmax(group_output, dim=1)
            
            # 第二级：根据第一级的结果，使用对应的细分模型
            final_pred = torch.zeros_like(group_pred)
            
            # 处理低唤醒组 (group_pred == 0)
            low_mask = (group_pred == 0)
            if low_mask.sum() > 0:
                low_output = self.low_arousal_model(x[low_mask])
                low_subpred = torch.argmax(low_output, dim=1)
                # 0-平静, 1-悲伤
                final_pred[low_mask] = low_subpred
            
            # 处理高唤醒组 (group_pred == 1)
            high_mask = (group_pred == 1)
            if high_mask.sum() > 0:
                high_output = self.high_arousal_model(x[high_mask])
                high_subpred = torch.argmax(high_output, dim=1)
                # 2-愉悦, 3-焦虑
                final_pred[high_mask] = high_subpred + 2
            
            return group_pred, final_pred


def train_model(model, train_loader, criterion, optimizer, device, level_name=""):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch_data in train_loader:
        if level_name == "group":
            x, group_label, _, _ = batch_data
            labels = group_label
        elif level_name == "low_arousal":
            x, group_label, subgroup_label, _ = batch_data
            # 只处理低唤醒组的样本
            mask = (group_label == 0)
            if mask.sum() == 0:
                continue
            x = x[mask]
            labels = subgroup_label[mask]
        else:  # high_arousal
            x, group_label, subgroup_label, _ = batch_data
            # 只处理高唤醒组的样本
            mask = (group_label == 1)
            if mask.sum() == 0:
                continue
            x = x[mask]
            labels = subgroup_label[mask]
        
        x, labels = x.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(x)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    if total == 0:
        return 0, 0
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100 * correct / total
    return epoch_loss, epoch_acc


def evaluate_model(model, test_loader, criterion, device, level_name=""):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch_data in test_loader:
            if level_name == "group":
                x, group_label, _, _ = batch_data
                labels = group_label
            elif level_name == "low_arousal":
                x, group_label, subgroup_label, _ = batch_data
                mask = (group_label == 0)
                if mask.sum() == 0:
                    continue
                x = x[mask]
                labels = subgroup_label[mask]
            else:  # high_arousal
                x, group_label, subgroup_label, _ = batch_data
                mask = (group_label == 1)
                if mask.sum() == 0:
                    continue
                x = x[mask]
                labels = subgroup_label[mask]
            
            x, labels = x.to(device), labels.to(device)
            
            outputs = model(x)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    if total == 0:
        return 0, 0
    
    epoch_loss = running_loss / len(test_loader)
    epoch_acc = 100 * correct / total
    return epoch_loss, epoch_acc


# 主程序
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}\n")
    
    # 加载数据
    data_file = r'1D_CNN\李佳宜_consolidated.csv'
    print("加载数据...")
    dataset = EmotionDataset(data_file)
    
    # 划分训练集和测试集
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    
    print(f"\n训练集大小: {train_size}")
    print(f"测试集大小: {test_size}")
    
    # 创建数据加载器
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # 创建双层分类系统
    hierarchical_classifier = HierarchicalClassifier(dataset.max_length, device)
    
    # 训练参数
    num_epochs = 100
    learning_rate = 0.0005
    weight_decay = 1e-4
    
    # 优化器和损失函数
    group_optimizer = optim.Adam(hierarchical_classifier.group_model.parameters(), 
                                 lr=learning_rate, weight_decay=weight_decay)
    low_optimizer = optim.Adam(hierarchical_classifier.low_arousal_model.parameters(), 
                               lr=learning_rate, weight_decay=weight_decay)
    high_optimizer = optim.Adam(hierarchical_classifier.high_arousal_model.parameters(), 
                                lr=learning_rate, weight_decay=weight_decay)
    
    criterion = nn.CrossEntropyLoss()
    
    # 训练历史
    history = {
        'group': {'train_loss': [], 'train_acc': [], 'test_loss': [], 'test_acc': []},
        'low': {'train_loss': [], 'train_acc': [], 'test_loss': [], 'test_acc': []},
        'high': {'train_loss': [], 'train_acc': [], 'test_loss': [], 'test_acc': []}
    }
    
    print("\n" + "="*80)
    print("开始训练双层分类系统")
    print("="*80)
    
    for epoch in range(num_epochs):
        # 训练第一级模型（分组）
        train_loss_g, train_acc_g = train_model(
            hierarchical_classifier.group_model, train_loader, criterion, 
            group_optimizer, device, "group"
        )
        test_loss_g, test_acc_g = evaluate_model(
            hierarchical_classifier.group_model, test_loader, criterion, 
            device, "group"
        )
        
        # 训练第二级模型（低唤醒组内）
        train_loss_l, train_acc_l = train_model(
            hierarchical_classifier.low_arousal_model, train_loader, criterion, 
            low_optimizer, device, "low_arousal"
        )
        test_loss_l, test_acc_l = evaluate_model(
            hierarchical_classifier.low_arousal_model, test_loader, criterion, 
            device, "low_arousal"
        )
        
        # 训练第二级模型（高唤醒组内）
        train_loss_h, train_acc_h = train_model(
            hierarchical_classifier.high_arousal_model, train_loader, criterion, 
            high_optimizer, device, "high_arousal"
        )
        test_loss_h, test_acc_h = evaluate_model(
            hierarchical_classifier.high_arousal_model, test_loader, criterion, 
            device, "high_arousal"
        )
        
        # 记录历史
        history['group']['train_loss'].append(train_loss_g)
        history['group']['train_acc'].append(train_acc_g)
        history['group']['test_loss'].append(test_loss_g)
        history['group']['test_acc'].append(test_acc_g)
        
        history['low']['train_loss'].append(train_loss_l)
        history['low']['train_acc'].append(train_acc_l)
        history['low']['test_loss'].append(test_loss_l)
        history['low']['test_acc'].append(test_acc_l)
        
        history['high']['train_loss'].append(train_loss_h)
        history['high']['train_acc'].append(train_acc_h)
        history['high']['test_loss'].append(test_loss_h)
        history['high']['test_acc'].append(test_acc_h)
        
        if (epoch + 1) % 10 == 0:
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            print(f"  第一级(分组)    - Train: {train_acc_g:.2f}% | Test: {test_acc_g:.2f}%")
            print(f"  第二级(低唤醒)  - Train: {train_acc_l:.2f}% | Test: {test_acc_l:.2f}%")
            print(f"  第二级(高唤醒)  - Train: {train_acc_h:.2f}% | Test: {test_acc_h:.2f}%")
    
    # 保存模型
    torch.save(hierarchical_classifier.group_model.state_dict(), '李佳宜_hierarchical_group.pth')
    torch.save(hierarchical_classifier.low_arousal_model.state_dict(), '李佳宜_hierarchical_low.pth')
    torch.save(hierarchical_classifier.high_arousal_model.state_dict(), '李佳宜_hierarchical_high.pth')
    
    print("\n" + "="*80)
    print("训练完成！")
    print("模型已保存:")
    print("  - 李佳宜_hierarchical_group.pth (第一级)")
    print("  - 李佳宜_hierarchical_low.pth (第二级-低唤醒)")
    print("  - 李佳宜_hierarchical_high.pth (第二级-高唤醒)")
    
    # 评估整体性能
    print("\n评估整体分类性能...")
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch_data in test_loader:
            x, _, _, original_label = batch_data
            x = x.to(device)
            
            _, final_pred = hierarchical_classifier.predict(x)
            all_preds.extend(final_pred.cpu().numpy())
            
            # 转换原始标签为数字
            label_map = {'平静': 0, '悲伤': 1, '愉悦': 2, '焦虑': 3}
            numeric_labels = [label_map[l] for l in original_label]
            all_labels.extend(numeric_labels)
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # 计算整体准确率
    overall_acc = 100 * np.sum(all_preds == all_labels) / len(all_labels)
    print(f"\n整体准确率: {overall_acc:.2f}%")
    
    # 绘制混淆矩阵
    cm = confusion_matrix(all_labels, all_preds)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['平静', '悲伤', '愉悦', '焦虑'],
                yticklabels=['平静', '悲伤', '愉悦', '焦虑'])
    plt.title('Hierarchical Classification - Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('李佳宜_hierarchical_confusion_matrix.png', dpi=300)
    print("混淆矩阵已保存: 李佳宜_hierarchical_confusion_matrix.png")
    
    # 绘制训练曲线
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # 第一级模型
    axes[0, 0].plot(history['group']['train_loss'], label='Train')
    axes[0, 0].plot(history['group']['test_loss'], label='Test')
    axes[0, 0].set_title('Level 1: Group Classifier - Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    axes[1, 0].plot(history['group']['train_acc'], label='Train')
    axes[1, 0].plot(history['group']['test_acc'], label='Test')
    axes[1, 0].set_title('Level 1: Group Classifier - Accuracy')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Accuracy (%)')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # 第二级模型 - 低唤醒
    axes[0, 1].plot(history['low']['train_loss'], label='Train')
    axes[0, 1].plot(history['low']['test_loss'], label='Test')
    axes[0, 1].set_title('Level 2: Low Arousal Classifier - Loss')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    axes[1, 1].plot(history['low']['train_acc'], label='Train')
    axes[1, 1].plot(history['low']['test_acc'], label='Test')
    axes[1, 1].set_title('Level 2: Low Arousal Classifier - Accuracy')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Accuracy (%)')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    # 第二级模型 - 高唤醒
    axes[0, 2].plot(history['high']['train_loss'], label='Train')
    axes[0, 2].plot(history['high']['test_loss'], label='Test')
    axes[0, 2].set_title('Level 2: High Arousal Classifier - Loss')
    axes[0, 2].set_xlabel('Epoch')
    axes[0, 2].set_ylabel('Loss')
    axes[0, 2].legend()
    axes[0, 2].grid(True)
    
    axes[1, 2].plot(history['high']['train_acc'], label='Train')
    axes[1, 2].plot(history['high']['test_acc'], label='Test')
    axes[1, 2].set_title('Level 2: High Arousal Classifier - Accuracy')
    axes[1, 2].set_xlabel('Epoch')
    axes[1, 2].set_ylabel('Accuracy (%)')
    axes[1, 2].legend()
    axes[1, 2].grid(True)
    
    plt.tight_layout()
    plt.savefig('李佳宜_hierarchical_training_curves.png', dpi=300)
    print("训练曲线已保存: 李佳宜_hierarchical_training_curves.png")
    
    # 打印详细分类报告
    print("\n详细分类报告:")
    print(classification_report(all_labels, all_preds, 
                                target_names=['平静', '悲伤', '愉悦', '焦虑'],
                                digits=4))
