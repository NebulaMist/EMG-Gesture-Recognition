import numpy as np
from sklearn.decomposition import PCA

def my_rms(emg_window):
    return np.sqrt(np.mean(emg_window ** 2))


def get_rms(emg, window_len, step_len, fs):
    """
    使用滑动窗口方法从EMG数据中提取RMS特征。

    参数:
    - emg: 一个形状为 (Nsample, Nchannel) 的二维numpy数组，每列是一个EMG数据通道。
    - window_len: 滑动窗口的长度（以秒为单位）。
    - step_len: 滑动窗口的步长（以秒为单位）。
    - fs: 采样频率（Hz）。

    返回:
    - rms: 提取的RMS特征，形状为 (num_windows, Nchannel)。
    """
    window_sample = int(np.floor(window_len * fs))
    step_sample = int(np.floor(step_len * fs))
    Nsample, Nchannel = emg.shape

    # 计算窗口的数量
    num_windows = (Nsample - window_sample) // step_sample + 1
    rms_features = np.zeros((num_windows, Nchannel))

    # 在信号上滑动窗口并计算每个通道的RMS
    for i in range(0, Nsample - window_sample + 1, step_sample):
        window = emg[i:i + window_sample, :]  # 获取窗口（形状为 window_sample x Nchannel）
        rms_features[i // step_sample, :] = np.apply_along_axis(my_rms, 0, window)  # 计算每个通道的RMS

    return rms_features


# 假设 my_wl 在其他地方定义并计算给定窗口的波形长度
def my_wl(emg_window, fs):
    # 波形长度计算的占位符
    # 实际实现将取决于WL的具体算法。
    return np.sum(np.abs(np.diff(emg_window)))  # 示例计算（波形长度）

def get_wl(emg, window_len, step_len, fs):
    """
    使用滑动窗口方法从EMG数据中提取波形长度特征。
    
    参数:
    - emg: 一个形状为 (Nsample, Nchannel) 的二维numpy数组，每列是一个EMG数据通道。
    - window_len: 滑动窗口的长度（以秒为单位）。
    - step_len: 滑动窗口的步长（以秒为单位）。
    - fs: 采样频率（Hz）。
    
    返回:
    - wl: 提取的波形长度特征，形状为 (num_windows, Nchannel)。
    """
    window_sample = int(np.floor(window_len * fs))
    step_sample = int(np.floor(step_len * fs))
    Nsample, Nchannel = emg.shape
    
    # 计算窗口的数量
    num_windows = (Nsample - window_sample) // step_sample + 1
    wl_features = np.zeros((num_windows, Nchannel))
    
    # 在信号上滑动窗口并计算每个通道的波形长度
    for i in range(0, Nsample - window_sample + 1, step_sample):
        window = emg[i:i + window_sample, :]  # 获取窗口（形状为 window_sample x Nchannel）
        wl_features[i // step_sample, :] = np.apply_along_axis(my_wl, 0, window, fs)
    
    return wl_features

def get_zc(emg, window_len, step_len, thresh, fs):
    """
    使用滑动窗口方法从EMG数据中提取零交叉特征。
    
    参数:
    - emg: 一个形状为 (Nsample, Nchannel) 的二维numpy数组，每列是一个EMG数据通道。
    - window_len: 滑动窗口的长度（以秒为单位）。
    - step_len: 滑动窗口的步长（以秒为单位）。
    - thresh: 用于检测有效零交叉的阈值。
    - fs: 采样频率（Hz）。
    
    返回:
    - zc: 提取的零交叉特征，形状为 (num_windows, Nchannel)。
    """
    window_sample = int(np.floor(window_len * fs))
    step_sample = int(np.floor(step_len * fs))
    Nsample, Nchannel = emg.shape

    # 计算窗口的数量
    num_windows = (Nsample - window_sample) // step_sample + 1
    zc_features = np.zeros((num_windows, Nchannel))
    
    fea_idx = 0
    for i in range(0, Nsample - window_sample + 1, step_sample):
        for j in range(Nchannel):
            emg_window = emg[i:i + window_sample, j]
            zc_features[fea_idx, j] = my_zc(emg_window, thresh)
        fea_idx += 1
    
    return zc_features
def my_zc(sig, thresh):
    """
    计算零交叉数量。
    
    参数：
    - sig: 输入的信号（1D数组）。
    - thresh: 检测有效零交叉的阈值。
    
    返回：
    - zc_value: 零交叉的数量。
    """
    N = len(sig)
    zc_value = 0
    for i in range(N - 1):
        if abs(sig[i + 1] - sig[i]) > thresh and sig[i] * sig[i + 1] < 0:
            zc_value += 1
    return zc_value


def get_ssc(emg, window_len, step_len, thresh, fs):
    """
    使用滑动窗口方法从EMG数据中提取斜率符号变化特征。
    
    参数:
    - emg: 一个形状为 (Nsample, Nchannel) 的二维numpy数组，每列是一个EMG数据通道。
    - window_len: 滑动窗口的长度（以秒为单位）。
    - step_len: 滑动窗口的步长（以秒为单位）。
    - thresh: 用于检测有效斜率符号变化的阈值。
    - fs: 采样频率（Hz）。
    
    返回:
    - ssc: 提取的斜率符号变化特征，形状为 (num_windows, Nchannel)。
    """
    window_sample = int(np.floor(window_len * fs))
    step_sample = int(np.floor(step_len * fs))
    Nsample, Nchannel = emg.shape

    # 计算窗口的数量
    num_windows = (Nsample - window_sample) // step_sample + 1
    ssc_features = np.zeros((num_windows, Nchannel))
    
    fea_idx = 0
    for i in range(0, Nsample - window_sample + 1, step_sample):
        for j in range(Nchannel):
            emg_window = emg[i:i + window_sample, j]
            ssc_features[fea_idx, j] = my_ssc(emg_window, thresh)
        fea_idx += 1
    
    return ssc_features

def my_ssc(sig, thresh):
    """
    计算斜率符号变化数量。
    
    参数：
    - sig: 输入的信号（1D数组）。
    - thresh: 检测有效斜率符号变化的阈值。
    
    返回：
    - ssc_value: 斜率符号变化的数量。
    """
    N = len(sig)
    ssc_value = 0
    for i in range(1, N - 1):
        if (sig[i] - sig[i - 1]) * (sig[i] - sig[i + 1]) > 0 and (
            abs(sig[i + 1] - sig[i]) > thresh or abs(sig[i - 1] - sig[i]) > thresh):
            ssc_value += 1
    return ssc_value



def feature_normalize(feature_train, feature_test, pca_active, dim):
    """
    归一化特征并可选地应用PCA降维。
    
    参数:
    - feature_train: 训练特征，形状为 (n_samples_train, n_features)
    - feature_test: 测试特征，形状为 (n_samples_test, n_features) 或 (n_features,)
    - pca_active: 是否使用PCA降维，1表示使用，0表示不使用
    - dim: PCA降维的目标维度
    
    返回:
    - feature_train_norm: 归一化（并可能降维）后的训练特征
    - feature_test_norm: 归一化（并可能降维）后的测试特征
    """
    print(f"\nInside feature_normalize:")
    print(f"Input shapes - Train: {feature_train.shape}, Test: {feature_test.shape}")
    
    # 确保测试特征是2D数组
    if feature_test.ndim == 1:
        feature_test = feature_test.reshape(1, -1)
        
    # 计算训练数据的均值和标准差（沿样本轴）
    mean_val = np.mean(feature_train, axis=0)
    std_val = np.std(feature_train, axis=0)
    
    # 避免除以零
    std_val[std_val == 0] = 1.0
    
    # 标准化处理
    feature_train_norm = (feature_train - mean_val) / std_val
    feature_test_norm = (feature_test - mean_val) / std_val
    
    # 应用PCA降维（如果激活）
    if pca_active:
        print(f"Using PCA with n_components={dim}")
        pca = PCA(n_components=dim)
        feature_train_norm = pca.fit_transform(feature_train_norm)  # (n_samples_train, dim)
        feature_test_norm = pca.transform(feature_test_norm)        # (n_samples_test, dim)
    
    print(f"归一化后 feature_train_norm shape: {feature_train_norm.shape}")
    print(f"归一化后 feature_test_norm shape: {feature_test_norm.shape}")
    
    return feature_train_norm, feature_test_norm
