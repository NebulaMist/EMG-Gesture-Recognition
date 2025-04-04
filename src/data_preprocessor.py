import numpy as np
import scipy.io as sio
import os
import tensorflow as tf
from tensorflow.keras import layers, models
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
        
        # 保存路径
        self.save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cnn_results")
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
    
    # 转换标签为one-hot编码
    num_classes = len(np.unique(labels))
    one_hot_labels = tf.keras.utils.to_categorical(labels - 1, num_classes)
    
    return normalized_data, one_hot_labels, num_classes