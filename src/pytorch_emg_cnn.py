import numpy as np
import scipy.io as sio
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import time

# CNN配置类
class CNNConfig:
    def __init__(self):
        # 数据路径
        self.data_path = "E:/11EMG/EMGdataset/gesture/"
        
        # 选择前10个受试者
        self.subjects = self.get_selected_subjects(10)
        
        # 任务参数
        self.task = "dynamic"
        
        # 窗口参数
        if self.task == "dynamic":
            self.window_len = 0.75
            self.step_len = 0.75
        else:
            self.window_len = 3.75
            self.step_len = 3.75
        
        # EMG参数
        self.fs_emg = 2048
        
        # CNN模型参数
        self.batch_size = 32
        self.epochs = 50
        self.learning_rate = 0.001
        self.validation_split = 0.2
        
        # PyTorch特定参数
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_workers = 0  # 数据加载线程数
        
        # 保存路径
        self.save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pytorch_results")
        os.makedirs(self.save_dir, exist_ok=True)
        
        # 时间戳
        self.timestamp = time.strftime("%Y%m%d-%H%M%S")
    
    def get_all_subjects(self):
        try:
            subjects = [d for d in os.listdir(self.data_path) 
                      if os.path.isdir(os.path.join(self.data_path, d))]
            return subjects
        except Exception as e:
            print(f"读取受试者目录时出错: {e}")
            return ["CQ1"]
    
    def get_selected_subjects(self, n=10):
        """获取前n个受试者"""
        all_subjects = self.get_all_subjects()
        return all_subjects[:n]  # 取前n个

# 数据加载函数 - 与Loader.py类似
def load_raw_emg_data(config):
    """加载前n个受试者的原始EMG数据"""
    all_emg_data = []
    all_labels = []
    all_subject_ids = []
    
    for subject in config.subjects:
        print(f"\n加载受试者 {subject} 的数据...")
        data_path = os.path.join(config.data_path, subject)
        
        try:
            # 加载EMG数据和标签
            data = sio.loadmat(os.path.join(data_path, "preprocessed_pr_motion.mat"))
            labels = sio.loadmat(os.path.join(data_path, "label_motion.mat"))
            
            emg_data = data['preprocessed_pr_motion']
            emg_labels = labels['label_motion'].flatten()
            
            Ntrial = emg_data[0].shape[0]
            Nsample = int(np.ceil(config.window_len * config.fs_emg))
            
            # 提取并处理每个试验的数据
            for j in range(Ntrial):
                # 获取最后Nsample个样本点
                emg = emg_data[0][j][-Nsample:]
                
                # 检查数据形状
                if emg.shape[0] == Nsample and emg.shape[1] == 256:
                    all_emg_data.append(emg)
                    all_labels.append(emg_labels[j])
                    all_subject_ids.append(subject)
                else:
                    print(f"警告: 试验 {j} 的数据形状有问题: {emg.shape}")
            
            print(f"成功加载 {subject} 的 {Ntrial} 个试验")
            
        except Exception as e:
            print(f"加载 {subject} 数据时出错: {e}")
    
    # 转换为numpy数组
    all_emg_data = np.array(all_emg_data)
    all_labels = np.array(all_labels)
    all_subject_ids = np.array(all_subject_ids)
    
    print(f"\n总共加载了 {len(all_emg_data)} 个样本")
    print(f"EMG数据形状: {all_emg_data.shape}")
    print(f"标签形状: {all_labels.shape}")
    
    return all_emg_data, all_labels, all_subject_ids

def reshape_to_matrix(emg_data):
    """
    将256个通道的EMG数据重新组织为16x16矩阵
    
    参数:
    - emg_data: 形状为(n_samples, time_points, 256)的EMG数据
    
    返回:
    - reshaped_data: 形状为(n_samples, time_points, 16, 16)的重组数据
    """
    n_samples, time_points, _ = emg_data.shape
    
    # 转换成16x16矩阵
    reshaped_data = np.zeros((n_samples, time_points, 16, 16))
    
    for i in range(n_samples):
        for t in range(time_points):
            # 从扁平的256通道重新组织为16x16矩阵
            reshaped_data[i, t] = emg_data[i, t].reshape(16, 16)
    
    return reshaped_data

def preprocess_for_cnn(emg_data, labels, config):
    """预处理EMG数据用于CNN模型"""
    # 首先转置数据以使通道维度在最后
    # 原始形状: (n_samples, time_points, channels)
    # 目标形状: (n_samples, time_points, 16, 16)
    
    # 假设原始数据形状为(n_samples, time_points, 256)
    n_samples, time_points, channels = emg_data.shape
    
    if channels != 256:
        print(f"警告: 通道数不是256，而是 {channels}")
        # 填充或截断到256通道
        if channels < 256:
            padding = np.zeros((n_samples, time_points, 256-channels))
            emg_data = np.concatenate([emg_data, padding], axis=2)
        else:
            emg_data = emg_data[:, :, :256]
    
    # 重组为16x16矩阵
    reshaped_data = reshape_to_matrix(emg_data)
    
    # 标准化处理
    mean_val = np.mean(reshaped_data, axis=(0, 1), keepdims=True)
    std_val = np.std(reshaped_data, axis=(0, 1), keepdims=True)
    std_val[std_val == 0] = 1.0  # 避免除以零
    normalized_data = (reshaped_data - mean_val) / std_val
    
    # 转换标签为one-hot编码 (PyTorch通常不需要，但保留以便比较)
    num_classes = len(np.unique(labels))
    # PyTorch不使用one-hot编码，而是使用整数索引，但我们仍然需要将标签从1开始调整为0开始
    adjusted_labels = labels - 1
    
    return normalized_data, adjusted_labels, num_classes

# PyTorch CNN模型定义
class EMGCNN(nn.Module):
    def __init__(self, input_channels, num_classes):
        super(EMGCNN, self).__init__()
        
        # 第一阶段：特征提取
        self.features = nn.Sequential(
            # 第一个卷积块
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # 第二个卷积块
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # 第三个卷积块
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # 第二阶段：分类
        # 计算经过三个MaxPool2d后的特征图大小 (16/2/2/2=2，因此是2x2)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(128 * 2 * 2, 256),  # 128个通道，2x2的特征图
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

def train_and_evaluate_cnn(X_train, y_train, X_test, y_test, config, num_classes):
    """训练并评估CNN模型 (PyTorch版本)"""
    print("创建PyTorch CNN模型...")
    
    # 调整输入形状为PyTorch CNN需要的格式
    # PyTorch期望的输入形状: (batch_size, channels, height, width)
    # 从 (n_samples, time_points, 16, 16) 转换为 (n_samples, time_points, 16, 16)
    # 然后转换为 (n_samples, time_points, 16, 16)
    
    # 如果时间点维度为1，则使用它作为通道维度
    if X_train.shape[1] == 1:
        # (samples, 1, 16, 16) -> (samples, 1, 16, 16)，不需要转置
        input_channels = 1
    else:
        # 多个时间点，将时间点作为通道
        # (samples, time_points, 16, 16) -> (samples, time_points, 16, 16)
        input_channels = X_train.shape[1]
    
    # 转换为PyTorch张量
    X_train_tensor = torch.from_numpy(X_train).float().permute(0, 1, 2, 3)
    y_train_tensor = torch.from_numpy(y_train).long()
    X_test_tensor = torch.from_numpy(X_test).float().permute(0, 1, 2, 3)
    y_test_tensor = torch.from_numpy(y_test).long()
    
    # 创建数据集
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    
    # 创建验证集
    train_size = int((1 - config.validation_split) * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        train_dataset, [train_size, val_size]
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.batch_size, 
        shuffle=True,
        num_workers=config.num_workers
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers
    )
    
    # 创建模型
    model = EMGCNN(input_channels, num_classes).to(config.device)
    print(model)
    
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    # 训练模型
    print("\n开始训练PyTorch CNN模型...")
    
    # 用于记录训练历史
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    # 最佳模型记录
    best_val_acc = 0.0
    best_model_path = os.path.join(config.save_dir, f"best_model_{config.timestamp}.pth")
    
    for epoch in range(config.epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(config.device), targets.to(config.device)
            
            # 梯度清零
            optimizer.zero_grad()
            
            # 前向传播
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # 反向传播
            loss.backward()
            optimizer.step()
            
            # 统计训练结果
            train_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            train_total += targets.size(0)
            train_correct += predicted.eq(targets).sum().item()
        
        train_loss = train_loss / train_total
        train_acc = train_correct / train_total
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(config.device), targets.to(config.device)
                
                # 前向传播
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                # 统计验证结果
                val_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                val_total += targets.size(0)
                val_correct += predicted.eq(targets).sum().item()
        
        val_loss = val_loss / val_total
        val_acc = val_correct / val_total
        
        # 更新学习率调度器
        scheduler.step(val_loss)
        
        # 记录历史
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # 打印进度
        print(f"Epoch {epoch+1}/{config.epochs} - "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_model_path)
            print(f"模型已保存至 {best_model_path}")
    
    # 加载最佳模型
    model.load_state_dict(torch.load(best_model_path))
    
    # 评估阶段
    print("\n在测试集上评估CNN模型...")
    model.eval()
    test_loss = 0.0
    test_correct = 0
    test_total = 0
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(config.device), targets.to(config.device)
            
            # 前向传播
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # 统计测试结果
            test_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            test_total += targets.size(0)
            test_correct += predicted.eq(targets).sum().item()
    
    test_loss = test_loss / test_total
    test_acc = test_correct / test_total
    print(f"测试集准确率: {test_acc:.4f}")
    
    return model, history, test_acc

def process_and_save_subject_data(subject, config):
    """单独处理每个受试者的数据并保存为预处理文件"""
    print(f"\n处理受试者 {subject} 的数据...")
    data_path = os.path.join(config.data_path, subject)
    save_path = os.path.join(config.save_dir, f"{subject}_processed.npz")
    
    # 如果已经处理过，则跳过
    if os.path.exists(save_path):
        print(f"已存在处理好的数据文件: {save_path}，跳过处理")
        return True
    
    try:
        # 加载EMG数据和标签
        data = sio.loadmat(os.path.join(data_path, "preprocessed_pr_motion.mat"))
        labels = sio.loadmat(os.path.join(data_path, "label_motion.mat"))
        
        emg_data = data['preprocessed_pr_motion']
        emg_labels = labels['label_motion'].flatten()
        
        Ntrial = emg_data[0].shape[0]
        Nsample = int(np.ceil(config.window_len * config.fs_emg))
        
        subject_emg_data = []
        subject_labels = []
        
        # 提取并处理每个试验的数据
        for j in range(Ntrial):
            # 获取最后Nsample个样本点
            emg = emg_data[0][j][-Nsample:]
            
            # 检查数据形状
            if emg.shape[0] == Nsample and emg.shape[1] == 256:
                subject_emg_data.append(emg)
                subject_labels.append(emg_labels[j])
            else:
                print(f"警告: 试验 {j} 的数据形状有问题: {emg.shape}")
        
        # 转换为numpy数组
        subject_emg_data = np.array(subject_emg_data)
        subject_labels = np.array(subject_labels)
        
        # 重组为16x16矩阵并标准化
        n_samples, time_points, channels = subject_emg_data.shape
        
        if channels != 256:
            print(f"警告: 通道数不是256，而是 {channels}")
            # 填充或截断到256通道
            if channels < 256:
                padding = np.zeros((n_samples, time_points, 256-channels))
                subject_emg_data = np.concatenate([subject_emg_data, padding], axis=2)
            else:
                subject_emg_data = subject_emg_data[:, :, :256]
        
        # 重组为16x16矩阵
        reshaped_data = reshape_to_matrix(subject_emg_data)
        
        # 标准化处理
        mean_val = np.mean(reshaped_data, axis=(0, 1), keepdims=True)
        std_val = np.std(reshaped_data, axis=(0, 1), keepdims=True)
        std_val[std_val == 0] = 1.0  # 避免除以零
        normalized_data = (reshaped_data - mean_val) / std_val
        
        # 调整标签从1开始到0开始
        adjusted_labels = subject_labels - 1
        
        # 保存处理后的数据
        np.savez(save_path, 
                 data=normalized_data,
                 labels=adjusted_labels)
        
        print(f"已保存处理好的数据 {subject}: {normalized_data.shape}")
        
        # 释放内存
        del subject_emg_data, reshaped_data, normalized_data
        return True
        
    except Exception as e:
        print(f"处理 {subject} 数据时出错: {e}")
        return False

# 自定义Dataset类来从保存的文件加载数据
class EMGDataset(torch.utils.data.Dataset):
    def __init__(self, subject_list, data_dir, transform=None):
        self.subject_list = subject_list
        self.data_dir = data_dir
        self.transform = transform
        
        # 加载所有文件的元数据（不加载实际数据）
        self.metadata = []
        self.class_counts = {}
        
        for subject in subject_list:
            file_path = os.path.join(data_dir, f"{subject}_processed.npz")
            if os.path.exists(file_path):
                # 只加载数据形状和标签信息
                with np.load(file_path) as data:
                    labels = data['labels']
                    for idx, label in enumerate(labels):
                        self.metadata.append((subject, idx, int(label)))
                        # 统计类别
                        if int(label) not in self.class_counts:
                            self.class_counts[int(label)] = 0
                        self.class_counts[int(label)] += 1
        
        print(f"数据集加载完成，共 {len(self.metadata)} 个样本")
        print(f"类别分布: {self.class_counts}")
        
    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, idx):
        subject, sample_idx, label = self.metadata[idx]
        file_path = os.path.join(self.data_dir, f"{subject}_processed.npz")
        
        # 按需加载单个样本
        with np.load(file_path) as data:
            sample = data['data'][sample_idx]
        
        # 转换为tensor
        sample = torch.from_numpy(sample).float().permute(0, 1, 2)
        label = torch.tensor(label, dtype=torch.long)
        
        if self.transform:
            sample = self.transform(sample)
            
        return sample, label

# 修改主函数流程
def main():
    """主程序流程"""
    config = CNNConfig()
    
    print(f"使用设备: {config.device}")
    
    # 1. 获取所有受试者列表
    all_subjects = config.subjects
    print(f"找到 {len(all_subjects)} 个受试者: {all_subjects}")
    
    # 2. 逐个处理受试者数据
    processed_subjects = []
    for subject in all_subjects:
        if process_and_save_subject_data(subject, config):
            processed_subjects.append(subject)
    
    print(f"成功处理 {len(processed_subjects)} 个受试者的数据")
    
    # 3. 创建数据集和加载器
    full_dataset = EMGDataset(processed_subjects, config.save_dir)
    
    # 确定类别数量
    num_classes = max(full_dataset.class_counts.keys()) + 1
    print(f"数据集类别数: {num_classes}")
    
    # 4. 划分训练/验证/测试集
    dataset_size = len(full_dataset)
    test_size = int(0.2 * dataset_size)
    train_val_size = dataset_size - test_size
    train_size = int(0.8 * train_val_size)
    val_size = train_val_size - train_size
    
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        full_dataset, 
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # 5. 创建数据加载器
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.batch_size, 
        shuffle=True,
        num_workers=config.num_workers
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers
    )
    
    # 6. 训练模型（此处使用修改后的训练函数，使用DataLoader而不是全部数据）
    model, history, test_acc = train_cnn_with_loaders(
        train_loader, val_loader, test_loader, config, num_classes
    )
    
    # 余下的代码与原始main函数相同...
def train_cnn_with_loaders(train_loader, val_loader, test_loader, config, num_classes):
    
    
    
    print("创建PyTorch CNN模型...")
    
    # 获取一个批次样本，确定输入形状
    for inputs, _ in train_loader:
        input_channels = inputs.shape[1]  # 通道数
        break
    
    # 创建模型
    model = EMGCNN(input_channels, num_classes).to(config.device)
    print(model)
    
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    # 训练模型
    print("\n开始训练PyTorch CNN模型...")
    
    # 用于记录训练历史
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    # 最佳模型记录
    best_val_acc = 0.0
    best_model_path = os.path.join(config.save_dir, f"best_model_{config.timestamp}.pth")
    
    for epoch in range(config.epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(config.device), targets.to(config.device)
            
            # 梯度清零
            optimizer.zero_grad()
            
            # 前向传播
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # 反向传播
            loss.backward()
            optimizer.step()
            
            # 统计训练结果
            train_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            train_total += targets.size(0)
            train_correct += predicted.eq(targets).sum().item()
        
        train_loss = train_loss / train_total
        train_acc = train_correct / train_total
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(config.device), targets.to(config.device)
                
                # 前向传播
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                # 统计验证结果
                val_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                val_total += targets.size(0)
                val_correct += predicted.eq(targets).sum().item()
        
        val_loss = val_loss / val_total
        val_acc = val_correct / val_total
        
        # 更新学习率调度器
        scheduler.step(val_loss)
        
        # 记录历史
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # 打印进度
        print(f"Epoch {epoch+1}/{config.epochs} - "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_model_path)
            print(f"模型已保存至 {best_model_path}")
    
    # 加载最佳模型
    model.load_state_dict(torch.load(best_model_path))
    
    # 评估阶段
    print("\n在测试集上评估CNN模型...")
    model.eval()
    test_loss = 0.0
    test_correct = 0
    test_total = 0
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(config.device), targets.to(config.device)
            
            # 前向传播
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # 统计测试结果
            test_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            test_total += targets.size(0)
            test_correct += predicted.eq(targets).sum().item()
    
    test_loss = test_loss / test_total
    test_acc = test_correct / test_total
    print(f"测试集准确率: {test_acc:.4f}")
    
    return model, history, test_acc

if __name__ == "__main__":
    main()

def train_cnn_with_loaders(train_loader, val_loader, test_loader, config, num_classes):
    """使用数据加载器训练CNN模型"""
    print("创建PyTorch CNN模型...")
    
    # 获取一个批次样本，确定输入形状
    for inputs, _ in train_loader:
        input_channels = inputs.shape[1]  # 通道数
        break
    
    # 创建模型
    model = EMGCNN(input_channels, num_classes).to(config.device)
    print(model)
    
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    # 训练模型
    print("\n开始训练PyTorch CNN模型...")
    
    # 用于记录训练历史
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    # 最佳模型记录
    best_val_acc = 0.0
    best_model_path = os.path.join(config.save_dir, f"best_model_{config.timestamp}.pth")
    
    for epoch in range(config.epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(config.device), targets.to(config.device)
            
            # 梯度清零
            optimizer.zero_grad()
            
            # 前向传播
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # 反向传播
            loss.backward()
            optimizer.step()
            
            # 统计训练结果
            train_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            train_total += targets.size(0)
            train_correct += predicted.eq(targets).sum().item()
        
        train_loss = train_loss / train_total
        train_acc = train_correct / train_total
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(config.device), targets.to(config.device)
                
                # 前向传播
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                # 统计验证结果
                val_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                val_total += targets.size(0)
                val_correct += predicted.eq(targets).sum().item()
        
        val_loss = val_loss / val_total
        val_acc = val_correct / val_total
        
        # 更新学习率调度器
        scheduler.step(val_loss)
        
        # 记录历史
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # 打印进度
        print(f"Epoch {epoch+1}/{config.epochs} - "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_model_path)
            print(f"模型已保存至 {best_model_path}")
    
    # 加载最佳模型
    model.load_state_dict(torch.load(best_model_path))
    
    # 评估阶段
    print("\n在测试集上评估CNN模型...")
    model.eval()
    test_loss = 0.0
    test_correct = 0
    test_total = 0
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(config.device), targets.to(config.device)
            
            # 前向传播
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # 统计测试结果
            test_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            test_total += targets.size(0)
            test_correct += predicted.eq(targets).sum().item()
    
    test_loss = test_loss / test_total
    test_acc = test_correct / test_total
    print(f"测试集准确率: {test_acc:.4f}")
    
    return model, history, test_acc