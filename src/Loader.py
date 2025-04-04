import numpy as np
import scipy.io as sio
import os
import time
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import accuracy_score, confusion_matrix
from LDA_feature_extractor import (
    get_rms, get_wl, get_zc, get_ssc, feature_normalize
)
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# 配置部分 - 便于修改参数
class Config:
    def __init__(self):
        # 数据路径
        self.data_path = "E:/11EMG/EMGdataset/gesture/"
        
        # 自动检测所有受试者目录
        self.subjects = self.get_all_subjects() 
        
        # 任务参数
        self.task = "dynamic"  # "dynamic" 或 "static"
        
        # 窗口参数
        if self.task == "dynamic":
            self.window_len = 0.75
            self.step_len = 0.75
        else:
            self.window_len = 3.75
            self.step_len = 3.75
            
        # 特征提取参数
        self.zc_ssc_thresh = 0.0004
        self.fs_emg = 2048
        
        # 降维参数
        self.pca_active = 1
        self.dim = 1000  # 固定为1000维，接近原始的1024维
        
        # 训练参数
        self.test_size = 0.2  # 测试集比例
        self.random_state = 42
        
        # 保存路径
        self.save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
        os.makedirs(self.save_dir, exist_ok=True)
        
        # 时间戳（用于结果文件命名）
        self.timestamp = time.strftime("%Y%m%d-%H%M%S")
        
    def get_all_subjects(self):
        """获取数据目录下的所有受试者文件夹"""
        try:
            subjects = [d for d in os.listdir(self.data_path) 
                       if os.path.isdir(os.path.join(self.data_path, d))]
            return subjects
        except Exception as e:
            print(f"读取受试者目录时出错: {e}")
            return ["CQ1"]  # 默认返回CQ1

# 1. 数据加载模块
def load_data(config, subject):
    """加载EMG数据和标签"""
    data_path = os.path.join(config.data_path, subject)
    
    try:
        data = sio.loadmat(os.path.join(data_path, "preprocessed_pr_motion.mat"))
        labels = sio.loadmat(os.path.join(data_path, "label_motion.mat"))
        
        emg_data = data['preprocessed_pr_motion']
        emg_labels = labels['label_motion'].flatten()
        
        return emg_data, emg_labels
    except Exception as e:
        print(f"加载{subject}数据出错: {e}")
        return None, None

# 2. 特征提取模块
def extract_features(emg_data, config):
    """从EMG数据中提取特征"""
    print(f"正在提取特征...")
    Nsample = int(np.ceil(config.window_len * config.fs_emg))
    Ntrial = emg_data[0].shape[0]
    
    all_features = []
    for j in range(Ntrial):
        emg = emg_data[0][j][-Nsample:]
        
        # 特征提取
        rms_tmp = get_rms(emg, config.window_len, config.step_len, config.fs_emg)
        wl_tmp = get_wl(emg, config.window_len, config.step_len, config.fs_emg)
        zc_tmp = get_zc(emg, config.window_len, config.step_len, config.zc_ssc_thresh, config.fs_emg)
        ssc_tmp = get_ssc(emg, config.window_len, config.step_len, config.zc_ssc_thresh, config.fs_emg)
        
        # 展平特征
        rms_flat = rms_tmp.flatten()
        wl_flat = wl_tmp.flatten()
        zc_flat = zc_tmp.flatten()
        ssc_flat = ssc_tmp.flatten()
        
        # 检查特征维度
        print(f"  试验 {j} 特征维度: RMS={len(rms_flat)}, WL={len(wl_flat)}, ZC={len(zc_flat)}, SSC={len(ssc_flat)}")
        
        # 合并特征
        trial_features = np.hstack([rms_flat, wl_flat, zc_flat, ssc_flat])
        
        # 确保特征维度是1024
        if len(trial_features) != 1024:
            print(f"  警告: 试验 {j} 特征维度 ({len(trial_features)}) 不是1024")
        
        all_features.append(trial_features)
    
    features = np.vstack(all_features)  # (Ntrial, feature_dim)
    
    print(f"特征提取完成，特征维度: {features.shape}")
    return features

# 3. 特征保存与加载模块
def save_combined_features(all_features, all_labels, all_subject_ids, config):
    """保存所有受试者的合并特征和标签"""
    save_path = os.path.join(config.save_dir, f"combined_features_{config.timestamp}.npz")
    np.savez(save_path, 
             features=all_features, 
             labels=all_labels, 
             subject_ids=all_subject_ids)
    print(f"合并特征已保存至 {save_path}")
    
def load_combined_features(config):
    """加载合并的特征和标签"""
    # 查找最近的特征文件
    feature_files = [f for f in os.listdir(config.save_dir) if f.startswith("combined_features_") and f.endswith(".npz")]
    if not feature_files:
        return None, None, None
    
    # 按修改时间排序，获取最新的文件
    latest_file = sorted(feature_files, key=lambda x: os.path.getmtime(os.path.join(config.save_dir, x)), reverse=True)[0]
    save_path = os.path.join(config.save_dir, latest_file)
    
    data = np.load(save_path, allow_pickle=True)
    print(f"从 {save_path} 加载合并特征")
    return data['features'], data['labels'], data['subject_ids']
    
def save_features(features, labels, config, subject):
    """保存单个受试者的特征和标签"""
    save_path = os.path.join(config.save_dir, f"{subject}_features.npz")
    np.savez(save_path, features=features, labels=labels)
    print(f"{subject}的特征已保存至 {save_path}")
    
def load_features(config, subject):
    """加载单个受试者的特征和标签"""
    save_path = os.path.join(config.save_dir, f"{subject}_features.npz")
    if os.path.exists(save_path):
        data = np.load(save_path)
        print(f"从 {save_path} 加载{subject}的特征")
        return data['features'], data['labels']
    else:
        return None, None

# 4. 模型训练与评估模块
def train_and_evaluate_model(features, labels, config):
    """训练并评估模型（使用训练集/测试集拆分）"""
    print("\n使用训练/测试集拆分训练模型...")
    
    # 拆分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=config.test_size, 
        random_state=config.random_state, stratify=labels
    )
    
    print(f"训练集大小: {X_train.shape}, 测试集大小: {X_test.shape}")
    
    # 特征归一化和降维
    print("正在进行特征归一化和降维...")
    
    # 使用固定的较高维度
    dim = min(config.dim, X_train.shape[1])  # 不超过特征数
    print(f"使用固定维度PCA: {dim}")
    X_train_norm, X_test_norm = feature_normalize(X_train, X_test, config.pca_active, dim)
    
    # 训练模型
    print("正在训练LDA模型...")
    lda = LDA()
    lda.fit(X_train_norm, y_train)
    
    # 预测
    y_pred = lda.predict(X_test_norm)
    
    # 计算准确率
    accuracy = accuracy_score(y_test, y_pred)
    print(f"LDA模型测试集准确率: {accuracy:.4f}")
    
    # 计算混淆矩阵
    cm = confusion_matrix(y_test, y_pred)
    
    return accuracy, y_pred, y_test, cm, lda

# 5. 跨受试者验证（可选）
def cross_subject_validation(all_features, all_labels, all_subject_ids, config):
    """进行跨受试者验证（留一受试者交叉验证）"""
    print("\n进行跨受试者验证...")
    
    unique_subjects = np.unique(all_subject_ids)
    num_subjects = len(unique_subjects)
    
    accuracies = []
    confusion_matrices = []
    
    for i, test_subject in enumerate(unique_subjects):
        print(f"\n[{i+1}/{num_subjects}] 以 {test_subject} 为测试集")
        
        # 划分训练集和测试集
        test_mask = (all_subject_ids == test_subject)
        train_mask = ~test_mask
        
        X_train = all_features[train_mask]
        y_train = all_labels[train_mask]
        X_test = all_features[test_mask]
        y_test = all_labels[test_mask]
        
        print(f"训练集大小: {X_train.shape}, 测试集大小: {X_test.shape}")
        
        # 特征归一化和降维 - 使用固定维度
        dim = min(config.dim, X_train.shape[1])
        print(f"使用固定维度PCA: {dim}")
        X_train_norm, X_test_norm = feature_normalize(X_train, X_test, config.pca_active, dim)
        
        # 训练模型
        lda = LDA()
        lda.fit(X_train_norm, y_train)
        
        # 预测
        y_pred = lda.predict(X_test_norm)
        
        # 计算准确率
        accuracy = accuracy_score(y_test, y_pred)
        print(f"受试者 {test_subject} 的准确率: {accuracy:.4f}")
        
        # 计算混淆矩阵
        cm = confusion_matrix(y_test, y_pred)
        
        accuracies.append(accuracy)
        confusion_matrices.append(cm)
    
    # 计算平均准确率
    mean_accuracy = np.mean(accuracies)
    print(f"\n跨受试者平均准确率: {mean_accuracy:.4f}")
    
    return accuracies, confusion_matrices

# 6. 结果可视化模块
def plot_confusion_matrix(cm, class_names, config, title='混淆矩阵'):
    """绘制混淆矩阵"""
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()
    
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)
    
    # 在格子中显示数值
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.ylabel('真实标签')
    plt.xlabel('预测标签')
    
    save_path = os.path.join(config.save_dir, f"cm_{config.timestamp}.png")
    plt.savefig(save_path)
    plt.close()
    print(f"混淆矩阵已保存至 {save_path}")

# 7. 特征处理主函数
def process_features(config):
    """处理所有受试者的特征"""
    # 尝试加载已有的合并特征
    all_features, all_labels, all_subject_ids = load_combined_features(config)
    
    if all_features is None:
        print("\n未找到已保存的合并特征，开始处理原始数据...")
        
        selected_subjects = []
        # 显示找到的所有受试者并让用户选择
        print(f"发现 {len(config.subjects)} 个受试者: {config.subjects}")
        
        use_all = input("是否使用所有受试者的数据? (y/n, 默认n): ").lower() == 'y'
        if not use_all:
            # 让用户选择要使用的受试者
            print("请选择要使用的受试者（输入序号，多个序号用空格分隔）:")
            for i, subject in enumerate(config.subjects):
                print(f"{i+1}. {subject}")
            
            choices = input("输入序号: ").split()
            try:
                indices = [int(c) - 1 for c in choices]
                selected_subjects = [config.subjects[i] for i in indices if 0 <= i < len(config.subjects)]
            except:
                print("输入格式错误，将只使用CQ1")
                selected_subjects = ["CQ1"]
        else:
            selected_subjects = config.subjects
            
        print(f"\n将处理以下受试者的数据: {selected_subjects}")
        
        # 处理每个受试者的数据并合并
        all_features = []
        all_labels = []
        all_subject_ids = []
        
        for subject in selected_subjects:
            print(f"\n处理受试者: {subject}")
            
            # 尝试从缓存加载特征
            features, labels = load_features(config, subject)
            
            # 如果没有缓存，则从原始数据提取特征
            if features is None:
                print(f"从原始数据提取{subject}的特征...")
                emg_data, labels = load_data(config, subject)
                
                if emg_data is not None:
                    features = extract_features(emg_data, config)
                    save_features(features, labels, config, subject)
                else:
                    print(f"跳过{subject}，未找到数据")
                    continue
            
            # 打印每个受试者的特征形状
            print(f"{subject} 特征形状: {features.shape}")
            
            # 确保特征是正确的形状 (n_samples, n_features)
            # 如果发现形状是 (n_features, n_samples)，则需要转置
            if features.shape[1] != 1024 and features.shape[0] == 1024:
                print(f"警告: {subject} 的特征形状似乎被转置了，进行修正...")
                features = features.T
            
            # 检查特征维度是否为1024
            if features.shape[1] != 1024:
                print(f"警告: {subject} 的特征维度 ({features.shape[1]}) 与预期 (1024) 不符!")
                print("跳过该受试者以避免合并错误")
                continue
            
            # 添加到合并列表
            all_features.append(features)
            all_labels.append(labels)
            subject_ids = np.array([subject] * len(labels))
            all_subject_ids.append(subject_ids)
        
        # 打印所有要合并的特征形状
        print("\n要合并的特征形状:")
        for i, subject in enumerate(selected_subjects):
            if i < len(all_features):
                print(f"{subject}: {all_features[i].shape}")
        
        # 合并所有受试者的数据
        try:
            all_features = np.vstack(all_features)
            all_labels = np.concatenate(all_labels)
            all_subject_ids = np.concatenate(all_subject_ids)
            
            print(f"\n合并后的特征维度: {all_features.shape}")
            print(f"合并后的标签维度: {all_labels.shape}")
            
            # 保存合并的特征
            save_combined_features(all_features, all_labels, all_subject_ids, config)
        except ValueError as e:
            print(f"合并特征时出错: {e}")
            
            # 详细检查每个数组的形状
            print("\n详细检查每个特征数组:")
            for i, subj in enumerate(selected_subjects[:len(all_features)]):
                print(f"{subj}:")
                print(f"  形状: {all_features[i].shape}")
                print(f"  数据类型: {all_features[i].dtype}")
                print(f"  非NaN值数量: {np.sum(~np.isnan(all_features[i]))}")
                print(f"  包含无穷大值: {np.any(np.isinf(all_features[i]))}")
            
            # 尝试进行手动堆叠
            print("\n尝试手动堆叠特征...")
            valid_features = []
            valid_labels = []
            valid_subject_ids = []
            
            for i, (feat, lab, subj_id) in enumerate(zip(all_features, all_labels, all_subject_ids)):
                # 检查特征是否有效且维度正确
                if feat.shape[1] == 1024 and not np.any(np.isnan(feat)) and not np.any(np.isinf(feat)):
                    valid_features.append(feat)
                    valid_labels.append(lab)
                    valid_subject_ids.append(subj_id)
                    print(f"保留受试者 {selected_subjects[i]} 的数据")
                else:
                    print(f"排除受试者 {selected_subjects[i]} 的数据")
            
            if valid_features:
                all_features = np.vstack(valid_features)
                all_labels = np.concatenate(valid_labels)
                all_subject_ids = np.concatenate(valid_subject_ids)
                print(f"成功合并 {len(valid_features)} 个受试者的数据")
                print(f"合并后的特征维度: {all_features.shape}")
                
                # 保存合并的特征
                save_combined_features(all_features, all_labels, all_subject_ids, config)
            else:
                print("没有有效的特征数据可合并")
                # 使用第一个特征作为返回值（虽然可能有问题）
                all_features = all_features[0] if all_features else np.array([])
                all_labels = all_labels[0] if all_labels else np.array([])
                all_subject_ids = all_subject_ids[0] if all_subject_ids else np.array([])
    
    return all_features, all_labels, all_subject_ids

# 修改feature_normalize函数
def feature_normalize(feature_train, feature_test, pca_active, dim):
    """
    归一化特征并可选地应用PCA降维。
    """
    print(f"\nInside feature_normalize:")
    print(f"Input shapes - Train: {feature_train.shape}, Test: {feature_test.shape}")
    
    # 确保测试特征是2D数组
    if feature_test.ndim == 1:
        feature_test = feature_test.reshape(1, -1)
        
    # 计算训练数据的均值和标准差
    mean_val = np.mean(feature_train, axis=0)
    std_val = np.std(feature_train, axis=0)
    
    # 避免除以零
    std_val[std_val == 0] = 1.0
    
    # 标准化处理
    feature_train_norm = (feature_train - mean_val) / std_val
    feature_test_norm = (feature_test - mean_val) / std_val
    
    # 应用PCA降维
    if pca_active:
        print(f"使用PCA降维至 {dim} 个主成分")
        pca = PCA(n_components=dim)
        feature_train_norm = pca.fit_transform(feature_train_norm)
        feature_test_norm = pca.transform(feature_test_norm)
        
        # 显示保留的方差比例
        explained_var = np.sum(pca.explained_variance_ratio_)
        print(f"保留了 {explained_var:.2%} 的方差")
    
    print(f"归一化后 feature_train_norm shape: {feature_train_norm.shape}")
    print(f"归一化后 feature_test_norm shape: {feature_test_norm.shape}")
    
    return feature_train_norm, feature_test_norm

# 8. 主程序流程
def main():
    """主程序流程"""
    config = Config()
    
    # 处理特征
    all_features, all_labels, all_subject_ids = process_features(config)
    
    # 生成类别名称
    class_names = [str(i) for i in range(1, len(np.unique(all_labels))+1)]
    
    # 选择评估方式
    print("\n请选择评估方式:")
    print("1. 随机划分训练集/测试集")
    print("2. 跨受试者验证")
    print("3. 两种方式都进行")
    choice = input("输入序号 (默认1): ") or "1"
    
    if choice in ["1", "3"]:
        # 随机划分训练集/测试集
        accuracy, y_pred, y_test, cm, model = train_and_evaluate_model(all_features, all_labels, config)
        plot_confusion_matrix(cm, class_names, config, title=f"混淆矩阵 - 随机划分 (准确率: {accuracy:.4f})")
        
        # 保存模型和结果
        results = {
            'method': 'random_split',
            'accuracy': accuracy,
            'predictions': y_pred,
            'true_labels': y_test,
            'confusion_matrix': cm,
        }
        np.savez(os.path.join(config.save_dir, f"random_split_results_{config.timestamp}.npz"), **results)
    
    if choice in ["2", "3"]:
        # 跨受试者验证
        accuracies, cms = cross_subject_validation(all_features, all_labels, all_subject_ids, config)
        
        # 保存结果
        results = {
            'method': 'cross_subject',
            'accuracies': accuracies,
            'confusion_matrices': cms,
            'mean_accuracy': np.mean(accuracies),
            'subject_ids': np.unique(all_subject_ids)
        }
        np.savez(os.path.join(config.save_dir, f"cross_subject_results_{config.timestamp}.npz"), **results)
        
        # 绘制平均混淆矩阵
        if len(cms) > 0:
            mean_cm = np.mean(cms, axis=0).astype(int)
            plot_confusion_matrix(mean_cm, class_names, config, 
                                 title=f"平均混淆矩阵 - 跨受试者 (准确率: {np.mean(accuracies):.4f})")

if __name__ == "__main__":
    main()