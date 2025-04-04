# EMG手势识别系统

基于表面肌电图(EMG)信号的手势识别系统，实现了三种不同的训练模式：
1. 单受试者LDA分类器
2. 多受试者LDA分类器
3. 基于CNN的多受试者深度学习模型

## 项目结构

```
EMG-Gesture-Recognition/
├── src/                   # 源代码
│   ├── loader2.py         # 单受试者LDA模型
│   ├── Loader.py          # 多受试者LDA模型
│   ├── CNN_RF_model_trainer.py  # CNN深度学习模型
│   ├── LDA_feature_extractor.py # 特征提取模块
│   ├── data_preprocessor.py     # 数据预处理模块
│   └── visualize_emg_matrices.py # EMG矩阵可视化
├── models/                # 保存训练好的模型
├── results/               # 实验结果
├── data/                  # 数据存放位置（不包含在仓库中）
├── docs/                  # 文档
└── README.md              # 项目说明文档
```

## 安装依赖

```bash
pip install numpy scipy scikit-learn tensorflow matplotlib
```

## 使用方法

### 单受试者LDA模型

```bash
python src/loader2.py
```

### 多受试者LDA模型

```bash
python src/Loader.py
```

### CNN模型训练

```bash
python src/CNN_RF_model_trainer.py
```

### EMG矩阵可视化

```bash
python src/visualize_emg_matrices.py --data_dir "path/to/data" --output_dir "path/to/output" --subjects "all" --animations
```
