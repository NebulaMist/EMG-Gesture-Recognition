import numpy as np
import scipy.io as sio
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import accuracy_score
from LDA_feature_extractor import (
    get_rms,
    get_wl,
    get_zc,
    get_ssc,
    feature_normalize
)
from sklearn.decomposition import PCA
import os  # 添加这行在文件开头

# 加载数据
data = sio.loadmat("E:/11EMG/EMGdataset/gesture/CQ1/preprocessed_pr_motion.mat")
labels = sio.loadmat("E:/11EMG/EMGdataset/gesture/CQ1/label_motion.mat")

data = data['preprocessed_pr_motion']
label = labels['label_motion'].flatten()

task = 'dynamic'

if task == 'dynamic':
    window_len = 0.75
    step_len = 0.75
else:
    window_len = 3.75
    step_len = 3.75

zc_ssc_thresh = 0.0004
fs_emg = 2048
pca_active = 1
dim = 200

Nsample = int(np.ceil(window_len * fs_emg))
Ntrial = data[0].shape[0]
predict_label[j] = lda.predict(feature_test_norm)[0]  # 提取第一个元素
all_features = []
for j in range(Ntrial):
    emg = data[0][j][-Nsample:]
    # 特征提取
    rms_tmp = get_rms(emg, window_len, step_len, fs_emg)
    wl_tmp = get_wl(emg, window_len, step_len, fs_emg)
    zc_tmp = get_zc(emg, window_len, step_len, zc_ssc_thresh, fs_emg)
    ssc_tmp = get_ssc(emg, window_len, step_len, zc_ssc_thresh, fs_emg)

    rms_flat = rms_tmp.flatten()
    wl_flat = wl_tmp.flatten()
    zc_flat = zc_tmp.flatten()
    ssc_flat = ssc_tmp.flatten()

    trial_features = np.hstack([rms_flat, wl_flat, zc_flat, ssc_flat])
    all_features.append(trial_features)

features = np.vstack(all_features)  # (Ntrial, feature_dim)
features = features.T  # 转置为 (feature_dim, Ntrial)

# 打印最终特征维度信息
print(f"Features shape: {features.shape}")

# 确保PCA维度合适
dim = min(dim, features.shape[0])  # PCA维度不超过特征数

# 留一法交叉验证
predict_label = np.zeros(Ntrial)
for j in range(Ntrial):
    # 取第j列为测试集
    feature_test = features[:, j]  # (feature_dim, )
    label_test = label[j]

    # 剩余列为训练集
    feature_train = np.delete(features, j, axis=1)  # (feature_dim, Ntrial-1)
    label_train = np.delete(label, j)

    # 归一化与PCA：feature_normalize需要(samples, features)的输入
    # 此时feature_train为(feature_dim, Ntrial-1)，转置后变(sampling, features)
    # 注：Ntrial-1为样本数，feature_dim为特征数
    print(f"归一化前 feature_train shape: {feature_train.T.shape}")
    print(f"归一化前 feature_test shape: {feature_test.T.shape}")
    
    feature_train_norm, feature_test_norm = feature_normalize(feature_train.T, feature_test.T, pca_active, dim)
    
    print(f"归一化后 feature_train_norm shape: {feature_train_norm.shape}")
    print(f"归一化后 feature_test_norm shape: {feature_test_norm.shape}")
    # 此时归一化后特征为(sampling, features)，可直接用于LDA

    # 如果需要加载保存的数据，可以这样做：
    # feature_train_norm = np.load('processed_data/feature_train_norm.npy')
    # feature_test_norm = np.load('processed_data/feature_test_norm.npy')
    # label_train = np.load('processed_data/label_train.npy')

    # 继续 LDA 训练
    print(f"Loading saved data for LDA training...")
    print(f"feature_train_norm shape: {feature_train_norm.shape}")
    print(f"label_train shape: {label_train.shape}")

    lda = LDA()
    lda.fit(feature_train_norm, label_train)
    predict_label[j] = lda.predict(feature_test_norm)

# 计算准确率
accuracy = accuracy_score(label, predict_label)
print(f"Accuracy: {accuracy}")
