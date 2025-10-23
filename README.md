# HRV情绪识别系统

基于心率变异性(HRV)特征的情绪识别系统，使用机器学习方法对四种情绪（开心、悲伤、平静、焦虑）进行分类预测。

## 📋 项目概述

本项目是一个完整的HRV情绪识别系统，包含数据预处理、特征提取、模型训练、评估和部署的完整流程。系统通过分析心率变异性数据，提取6个核心HRV特征，使用支持向量机(SVM)进行情绪分类。

### 主要功能
- **HRV特征提取**: 从心率数据中提取6个核心HRV特征
- **情绪分类**: 支持四种情绪的分类（开心、悲伤、平静、焦虑）
- **模型训练**: 提供SVM四分类和两两分类两种训练模式
- **模型评估**: 完整的模型性能评估和可视化
- **ONNX导出**: 支持将训练好的模型导出为ONNX格式，便于部署

## 🏗️ 项目结构

```
emotion/
├── data/                           # 原始心率数据
│   ├── 804愉悦平静组/              # 按情绪分组的原始数据
│   ├── 805/                        # 个人数据文件夹
│   └── ...
├── individual_data/                # 个人数据分析
│   ├── data/                       # 个人HRV数据
│   ├── hrv_emotion_analysis.py     # 个人情绪分析
│   └── delta_feature_analysis.py   # 特征差异分析
├── svm/                           # SVM模型相关
│   ├── train.py                   # SVM训练脚本
│   └── *.csv                      # 训练数据
├── onnx_models/                   # ONNX模型文件
├── processed_data/                # 处理后的数据
├── special_data/                  # 特殊数据
├── 40s_data/                      # 40秒数据 包含了40s处理流程的相关脚本 也包含了10s数据的脚本
├── done_data/                     # 已完成处理的数据
├── forMK/                         # MK相关数据
├── to_watch/                      # 与手表端协作
├── hrv_feature_extractor.py       
├── hrv_feature_extractor_with_quality.py  
├── svm_four_class_trainer.py     # SVM四分类训练器
├── svm_pairwise_classifier.py    # SVM两两分类器
├── run_svm_training.py           # 快速SVM训练脚本
├── run_svm_with_onnx.py          # SVM训练和ONNX导出
├── csv_process.py               # CSV数据处理
├── process_any_csv.py           # 通用CSV处理
├── fengzhijiance.py             # 峰值检测工具
├── formatter.py                 # 数据格式化工具
├── cleaning.py                  # 数据清洗工具
└── README.md                  
```

## 🚀 快速开始

### 环境要求

```bash
Python >= 3.9
numpy
pandas
scikit-learn
matplotlib
seaborn
scipy
joblib
```

可选依赖（用于ONNX导出）：
```bash
skl2onnx
onnx
```

### 安装依赖

```bash
pip install numpy pandas scikit-learn matplotlib seaborn scipy joblib
```

如需ONNX支持：
```bash
pip install skl2onnx onnx
```

## 📝 使用示例

### 完整工作流程示例

```bash
# 1. 处理原始数据
python ./40s_data/splitter.py

# 2. 提取HRV特征
python ./40s_data/processor_batch_40s.py 
# 根据数据长度调整对应参数，该脚本整合了峰值检测与特征计算 

# 3. 训练SVM模型
python svm_pairwise_classifier.py
# 可直接在该脚本导出onnx格式的svm模型
```

## 📊 HRV特征说明

系统提取的6个核心HRV特征：

| 特征名称 | 描述 | 计算方式 |
|---------|------|----------|
| RMSSD | 相邻RR间期差值的均方根 | √(Σ(RR_i - RR_{i+1})² / n) |
| pNN58 | 相邻RR间期差值超过58ms的百分比 | (|RR_i - RR_{i+1}| > 58ms) / n × 100% |
| SDNN | RR间期标准差 | √(Σ(RR_i - RR_mean)² / (n-1)) |
| SD1 | Poincaré图的短轴标准差 | √(Var(RR_diff) / 2) |
| SD2 | Poincaré图的长轴标准差 | √(2 × Var(RR) - Var(RR_diff) / 2) |
| SD1_SD2 | SD1与SD2的比值 | SD1 / SD2 |

## 🎯 情绪分类

系统支持四种情绪的分类：

- **开心**: 愉悦、小开心、愉悦等
- **悲伤**: 悲伤、大悲、小悲、悲等  
- **平静**: 平静状态
- **焦虑**: 焦虑、紧张等

## 🔧 主要工具

### SVM四分类训练器

**文件**: `svm_four_class_trainer.py`

- 完整的SVM四分类训练流程
- 超参数调优
- 模型评估和可视化
- 支持模型保存和加载

**功能特点**:
- 网格搜索优化参数
- 交叉验证评估
- 混淆矩阵可视化
- ROC曲线分析
- 特征重要性分析

### SVM两两分类器

**文件**: `svm_pairwise_classifier.py`

- 训练所有情绪对的两两分类器
- 支持ONNX模型导出
- 详细的性能分析

**训练模式**:
1. 训练所有SVM两两组合
2. 训练指定的SVM两两组合
3. 自动训练并导出ONNX模型

### 数据处理工具

#### CSV处理器 (`process_any_csv.py`)
- 处理情绪标签标准化
- 自动检测文件格式
- 统计情绪分布

#### 数据清洗器 (`cleaning.py`)
- 缺失值处理（中位数填充）
- 异常值检测和处理（IQR方法）
- 数据质量验证

#### 峰值检测器 (`fengzhijiance.py`)
- 可视化峰值检测结果
- 计算HRV特征
- 支持自定义时间窗口


## 📁 数据格式

### 输入数据格式

心率数据CSV文件应包含两列：
- 第1列：时间（毫秒）
- 第2列：心率信号值

### 输出数据格式

HRV特征CSV文件包含6列：
- RMSSD, pNN58, SDNN, SD1, SD2, SD1_SD2

训练数据CSV文件额外包含：
- emotion列：情绪标签

## ⚙️ 配置参数

### HRV特征提取参数

```python
# 数据段长度（毫秒）
SEGMENT_DURATION_MS = 40000  # 40秒

# 峰值检测参数
PEAK_DETECTION_PARAMS = {
    'distance': 5,           # 峰值间最小距离
    'prominence': 25,        # 峰值突出度
    'height': None           # 峰值高度阈值
}
```

## 🐛 故障排除

### 常见问题

1. **峰值检测失败**
   - 检查信号质量
   - 调整峰值检测参数
   - 确保数据长度足够

2. **模型训练失败**
   - 检查数据格式
   - 确保有足够的训练样本
   - 验证情绪标签正确性

3. **ONNX导出失败**
   - 安装skl2onnx和onnx包
   - 检查模型兼容性

### 调试工具

- 使用`fengzhijiance.py`可视化峰值检测
- 使用`cleaning.py`检查数据质量
- 查看训练日志和错误信息

## 📞 联系方式

如有问题或建议，请通过以下方式联系：

- 提交Issue
- 发送邮件
- 创建Pull Request

## 🙏 致谢

感谢所有为这个项目做出贡献的开发者和研究人员。

---

**注意**: 本项目仅用于学术研究目的，请确保在使用时遵守相关的伦理和隐私规定。