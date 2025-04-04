import os
import shutil
import sys

def organize_repository():
    """整理项目仓库结构，复制所有源代码文件到对应目录"""
    
    # 本地仓库路径
    repo_dir = os.path.abspath("e:/11EMG/SSSBProject/EMG-Gesture-Recognition")
    # 源代码目录
    src_dir = os.path.abspath("e:/11EMG/SSSBProject/SSSBProject2/src")
    
    # 确保目录存在
    if not os.path.exists(repo_dir):
        print(f"错误: 仓库目录 {repo_dir} 不存在!")
        return False
    
    if not os.path.exists(src_dir):
        print(f"错误: 源代码目录 {src_dir} 不存在!")
        return False
    
    # 创建仓库的目录结构
    directories = [
        "src",
        "models",
        "results",
        "data",
        "docs",
        "visualizations"
    ]
    
    for directory in directories:
        dir_path = os.path.join(repo_dir, directory)
        os.makedirs(dir_path, exist_ok=True)
        print(f"创建目录: {dir_path}")
    
    # 复制所有源代码文件到仓库的src目录
    repo_src_dir = os.path.join(repo_dir, "src")
    
    # 获取源目录中的所有Python文件
    py_files = [f for f in os.listdir(src_dir) if f.endswith('.py')]
    
    for file in py_files:
        src_file = os.path.join(src_dir, file)
        dst_file = os.path.join(repo_src_dir, file)
        
        # 复制文件
        shutil.copy2(src_file, dst_file)
        print(f"复制文件: {src_file} -> {dst_file}")
    
    # 创建README.md文件(如果不存在)
    readme_path = os.path.join(repo_dir, "README.md")
    if not os.path.exists(readme_path):
        with open(readme_path, "w", encoding="utf-8") as f:
            f.write("""# EMG手势识别系统

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
""")
        print(f"创建README.md文件: {readme_path}")
    
    # 创建.gitignore文件(如果不存在)
    gitignore_path = os.path.join(repo_dir, ".gitignore")
    if not os.path.exists(gitignore_path):
        with open(gitignore_path, "w", encoding="utf-8") as f:
            f.write("""# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
*.egg-info/
.installed.cfg
*.egg

# 虚拟环境
venv/
ENV/

# IDE相关
.idea/
.vscode/
*.swp
*.swo

# 数据集和大文件
data/*.mat
data/*.npz
*.npz
*.npy
*.h5
*.hdf5

# 结果目录中的大文件
results/*.npz
results/*.npy
visualizations/*.png
visualizations/*.gif

# 日志和输出文件
logs/
*.log
output/

# 系统文件
.DS_Store
Thumbs.db
""")
        print(f"创建.gitignore文件: {gitignore_path}")
    
    print("\n仓库结构整理完成!")
    return True

if __name__ == "__main__":
    organize_repository()
