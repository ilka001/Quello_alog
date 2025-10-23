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
├── 📁 data/                           # 原始心率数据目录
│   ├── 804愉悦平静组/                 # 愉悦和平静情绪组合的原始数据
│   │   ├── *.csv                      # 心率信号CSV文件（时间-数值格式）
│   │   └── *.txt                      # 情绪标签和时间范围文件
│   ├── 805/                           # 个人数据文件夹（805号被试）
│   │   ├── *.csv                      # 心率数据文件
│   │   └── *.txt                      # 标签文件
│   ├── 806悲伤平静组/                 # 悲伤和平静情绪组合数据
│   ├── 807/                           # 807号被试数据
│   ├── 808/                           # 808号被试数据（大量数据）
│   ├── 812/                           # 812号被试数据
│   ├── 814/                           # 814号被试数据
│   ├── 815/                           # 815号被试数据
│   ├── 818/                           # 818号被试数据（最大数据集）
│   ├── 819/                           # 819号被试数据
│   ├── 820/                           # 820号被试数据
│   ├── 821/                           # 821号被试数据
│   ├── 822/                           # 822号被试数据
│   └── PROBOLEM_DATA/                 # 问题数据（质量不佳的数据）
│       ├── *.CSV                     # 有问题的原始数据文件
│       └── *.txt                     # 问题描述文件
│
├── 📁 individual_data/                # 个人数据分析目录
│   ├── data/                          # 个人HRV特征数据
│   │   ├── *姓名_hrv_data.csv         # 个人HRV特征文件
│   │   └── ...                        # 100+个个人数据文件
│   ├── hrv_emotion_analysis.py        # 个人情绪分析脚本
│   ├── delta_feature_analysis.py      # 特征差异分析脚本
│   ├── hrv_classification_algorithms.py # 分类算法比较
│   ├── feature_optimization.py        # 特征优化脚本
│   ├── delta_results.csv             # 特征差异分析结果
│   └── median_delta_results.csv      # 中位数差异分析结果
│
├── 📁 svm/                           # SVM模型训练相关
│   ├── train.py                      # SVM训练脚本
│   ├── *.csv                         # 训练数据集
│   └── *.joblib                      # 训练好的模型文件
│
├── 📁 onnx_models/                   # ONNX模型文件目录
│   ├── *.onnx                        # 导出的ONNX模型文件
│   └── *_info.txt                    # 模型信息文件
│
├── 📁 processed_data/                # 处理后的数据目录
│   └── xx.bat                        # 批处理脚本
│
├── 📁 special_data/                  # 特殊数据集
│   ├── HOR/                          # HOR数据集
│   │   ├── hrv_MK.csv               # MK的HRV特征数据
│   │   ├── hrv_MK_train.csv         # MK训练数据
│   │   ├── hrv_MK_int8.csv          # MK的int8格式数据
│   │   ├── K.csv                    # K情绪原始数据
│   │   ├── F.csv                    # F情绪原始数据
│   │   ├── B.csv                    # B情绪原始数据
│   │   ├── P.csv                    # P情绪原始数据
│   │   ├── P2.csv, P3.csv           # P情绪变体数据
│   │   └── 0.txt                    # 标签文件
│   └── TKS/                          # TKS数据集
│       ├── hrv_FB.csv               # FB的HRV特征数据
│       ├── hrv_FB_train.csv         # FB训练数据
│       ├── hrv_FB_int8.csv          # FB的int8格式数据
│       ├── B/                        # B情绪数据子目录
│       │   └── 1分至18分钟 4分.csv   # 时间分段数据
│       ├── P/                        # P情绪数据子目录
│       │   ├── P1.csv, P2.csv       # P情绪分段数据
│       │   ├── P3.csv, P4.csv       # P情绪分段数据
│       └── 0.txt                    # 标签文件
│
├── 📁 40s_data/                      # 40秒数据处理流程
│   ├── splitter.py                   # 数据分割脚本
│   ├── processor_40s.py              # 40秒数据处理器
│   ├── processor_10s.py              # 10秒数据处理器
│   ├── processor_batch_40s.py       # 批量40秒数据处理器
│   ├── text_parser.py               # 文本解析器
│   ├── run.py                       # 运行脚本
│   ├── README.md                    # 40s数据处理说明
│   ├── LABEL.txt                    # 标签文件
│   ├── TIME_LABEL.txt               # 时间标签文件
│   ├── new.txt                      # 新标签文件
│   ├── backup_batch/                # 批量处理备份目录
│   │   └── *.csv                    # RR间隔数据备份文件
│   └── test/                        # 测试数据目录
│       ├── *.txt                    # 测试标签文件
│       └── 备注*.txt                # 测试备注文件
│
├── 📁 done_data/                     # 已完成处理的数据
│   ├── hrv_data_10s.csv             # 10秒HRV特征数据
│   ├── hrv_data_40s.csv             # 40秒HRV特征数据
│   ├── hrv_data_10s_cleaned.csv     # 清洗后的10秒数据
│   ├── hrv_data_10s_fortrain.csv    # 用于训练的10秒数据
│   ├── four_emotion.csv             # 四情绪分类数据
│   ├── tag.csv                      # 标签数据
│   └── README.md                    # 完成数据说明
│
├── 📁 forMK/                        # MK相关数据和可视化
│   ├── 1.csv, 2.csv, 3.csv          # MK的HRV特征数据文件
│   ├── 4.csv, 5.csv                 # MK的HRV特征数据文件
│   ├── fengzhijiance.py             # 峰值检测脚本
│   └── visualization/                # 可视化结果目录
│       ├── 1.png, 1-f.png           # 数据1的波形图和峰值图
│       ├── 2.png, 2-f.png           # 数据2的波形图和峰值图
│       ├── 3.png, 3-f.png           # 数据3的波形图和峰值图
│       ├── 4.png, 4-f.png           # 数据4的波形图和峰值图
│       └── 5.png, 5-f.png           # 数据5的波形图和峰值图
│
├── 📁 to_watch/                     # 与手表端协作相关
│   ├── test_hrv_simple.py          # 简单HRV测试脚本
│   ├── test_onnx_model.py          # ONNX模型测试脚本
│   ├── test_output_format.py       # 输出格式测试脚本
│   └── test_output_simple.py       # 简单输出测试脚本
│
├── 🔧 核心脚本文件
│   ├── hrv_feature_extractor.py     # HRV特征提取器（10秒版本）
│   ├── hrv_feature_extractor_with_quality.py # 带质量评估的HRV特征提取器
│   ├── svm_four_class_trainer.py    # SVM四分类训练器
│   ├── svm_pairwise_classifier.py   # SVM两两分类器
│   ├── run_svm_training.py          # 快速SVM训练脚本
│   ├── run_svm_with_onnx.py         # SVM训练和ONNX导出脚本
│   ├── csv_process.py              # CSV数据处理脚本
│   ├── process_any_csv.py          # 通用CSV处理脚本
│   ├── fengzhijiance.py            # 峰值检测工具
│   ├── formatter.py                # 数据格式化工具
│   ├── cleaning.py                 # 数据清洗工具
│   ├── csv_paths_list.txt          # CSV文件路径列表
│   └── clear_processed_data.bat     # 清理处理数据的批处理脚本
│
├── 📊 结果文件
│   ├── confusion_matrix.png         # 混淆矩阵图
│   ├── multiclass_roc_curve.png    # 多分类ROC曲线图
│   └── README.md                   # 项目说明文档
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