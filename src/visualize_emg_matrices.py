import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.colors import LinearSegmentedColormap
import argparse
from matplotlib.animation import FuncAnimation

def create_custom_colormap():
    """创建自定义颜色映射，从深蓝到红色"""
    colors = [(0, 0, 0.8), (0, 0.8, 0.8), (0.9, 0.9, 0.1), (0.8, 0.2, 0)]
    return LinearSegmentedColormap.from_list('EMG_cmap', colors)

def plot_emg_matrix(data, subject, sample_idx, time_idx, label, save_path=None):
    """绘制单个EMG 16x16矩阵热力图"""
    # 获取矩阵数据
    matrix = data[time_idx]
    
    # 创建自定义颜色映射
    cmap = create_custom_colormap()
    
    plt.figure(figsize=(10, 8))
    plt.imshow(matrix, cmap=cmap, interpolation='nearest')
    plt.colorbar(label='归一化EMG强度')
    
    # 设置标题和轴标签
    plt.title(f"受试者: {subject}, 样本: {sample_idx}, 类别: {label}, 时间点: {time_idx}")
    plt.xlabel("通道 X 方向")
    plt.ylabel("通道 Y 方向")
    
    # 添加网格
    plt.grid(False)
    
    # 设置轴刻度
    plt.xticks(np.arange(0, 16, 4))
    plt.yticks(np.arange(0, 16, 4))
    
    # 保存或显示
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.tight_layout()
        plt.show()

def create_animation(data, subject, sample_idx, label, save_path=None):
    """创建时间序列动画"""
    fig, ax = plt.subplots(figsize=(10, 8))
    cmap = create_custom_colormap()
    
    # 初始化图像
    matrix = data[0]
    img = ax.imshow(matrix, cmap=cmap, interpolation='nearest')
    plt.colorbar(img, ax=ax, label='归一化EMG强度')
    
    # 设置标题和轴标签
    title = ax.set_title(f"受试者: {subject}, 样本: {sample_idx}, 类别: {label}, 时间点: 0")
    ax.set_xlabel("通道 X 方向")
    ax.set_ylabel("通道 Y 方向")
    
    # 设置轴刻度
    ax.set_xticks(np.arange(0, 16, 4))
    ax.set_yticks(np.arange(0, 16, 4))
    
    def update(frame):
        """更新动画帧"""
        matrix = data[frame]
        img.set_array(matrix)
        title.set_text(f"受试者: {subject}, 样本: {sample_idx}, 类别: {label}, 时间点: {frame}")
        return [img, title]
    
    # 创建动画
    ani = FuncAnimation(fig, update, frames=len(data), interval=100, blit=True)
    
    # 保存或显示
    if save_path:
        ani.save(save_path, writer='pillow', fps=10)
        plt.close()
    else:
        plt.tight_layout()
        plt.show()
    
    return ani

def visualize_subject_data(subject, data_dir, output_dir, samples_per_class=1, time_points=None, create_animations=False):
    """可视化指定受试者的EMG矩阵数据"""
    # 构建文件路径
    file_path = os.path.join(data_dir, f"{subject}_processed.npz")
    
    # 如果精确匹配的文件不存在，尝试列出目录中所有文件并查找最接近的匹配
    if not os.path.exists(file_path):
        print(f"找不到受试者 {subject} 的数据文件: {file_path}")
        
        # 列出目录中所有文件
        all_files = os.listdir(data_dir)
        print(f"目录 {data_dir} 中的文件:")
        for f in all_files:
            print(f"  - {f}")
        
        # 尝试查找包含受试者名称的文件
        potential_files = [f for f in all_files if subject in f and f.endswith('.npz')]
        if potential_files:
            print(f"找到可能匹配的文件: {potential_files}")
            file_path = os.path.join(data_dir, potential_files[0])
            print(f"将使用: {file_path}")
        else:
            return False
    
    # 创建输出目录
    subject_output_dir = os.path.join(output_dir, subject)
    os.makedirs(subject_output_dir, exist_ok=True)
    
    # 加载数据
    print(f"正在加载 {subject} 的数据...")
    
    try:
        with np.load(file_path) as data:
            # 打印文件中的键，帮助调试
            print(f"文件 {file_path} 包含以下键: {list(data.keys())}")
            
            if 'data' not in data or 'labels' not in data:
                print(f"错误: 文件中没有'data'或'labels'键")
                return False
            
            matrix_data = data['data']
            labels = data['labels']
            
            print(f"加载的数据形状: {matrix_data.shape}")
            print(f"加载的标签形状: {labels.shape}")
    except Exception as e:
        print(f"加载数据时出错: {e}")
        return False
    
    # 获取类别信息
    unique_labels = np.unique(labels)
    print(f"发现 {len(unique_labels)} 个类别")
    
    # 对每个类别，选择样本
    for label in unique_labels:
        # 找到属于该类别的样本索引
        indices = np.where(labels == label)[0]
        
        # 如果样本数量不足，则使用所有样本
        n_samples = min(samples_per_class, len(indices))
        selected_indices = indices[:n_samples]
        
        for idx in selected_indices:
            # 获取要可视化的样本数据
            sample_data = matrix_data[idx]
            
            # 如果指定了时间点，则只选择那些时间点
            if time_points is not None:
                time_indices = [t for t in time_points if t < sample_data.shape[0]]
            else:
                # 默认选择第一个、中间和最后一个时间点
                time_indices = [0, sample_data.shape[0]//2, sample_data.shape[0]-1]
            
            # 为每个选择的时间点创建可视化
            for t in time_indices:
                save_path = os.path.join(
                    subject_output_dir, 
                    f"{subject}_sample{idx}_class{label}_time{t}.png"
                )
                plot_emg_matrix(sample_data, subject, idx, t, label, save_path)
            
            # 如果需要，创建动画
            if create_animations:
                animation_path = os.path.join(
                    subject_output_dir, 
                    f"{subject}_sample{idx}_class{label}_animation.gif"
                )
                create_animation(sample_data, subject, idx, label, animation_path)
    
    print(f"完成 {subject} 的可视化，图像保存在 {subject_output_dir}")
    return True

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='可视化EMG 16x16矩阵数据')
    parser.add_argument('--data_dir', default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "pytorch_results"), 
                        help='包含预处理数据的目录')
    parser.add_argument('--output_dir', default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "visualizations"), 
                        help='保存可视化结果的目录')
    parser.add_argument('--subjects', default='all', 
                        help='要可视化的受试者，逗号分隔或"all"表示所有受试者')
    parser.add_argument('--samples', type=int, default=2, 
                        help='每个类别要可视化的样本数量')
    parser.add_argument('--time_points', default=None, 
                        help='要可视化的时间点索引，逗号分隔或留空表示自动选择')
    parser.add_argument('--animations', action='store_true', 
                        help='是否创建动画')
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 确定要可视化的受试者
    if args.subjects.lower() == 'all':
        # 获取数据目录中的所有.npz文件并正确提取主题名
        subject_files = []
        for f in os.listdir(args.data_dir):
            if f.endswith('_processed.npz'):
                subject_name = f[:-12]  # 移除"_processed.npz"
                subject_files.append(subject_name)
        
        # 打印找到的文件
        print(f"在目录 {args.data_dir} 中找到以下预处理文件:")
        for f in subject_files:
            print(f"- {f}")
            
        subjects = subject_files
    else:
        subjects = [s.strip() for s in args.subjects.split(',')]
    
    # 如果没有找到任何文件，尝试特定的受试者名称
    if not subjects:
        default_subjects = ['CQ1', 'CQ2', 'DCY1', 'DCY2', 'DHK1', 'DHK2', 'DHQ1', 'DHQ2', 'DY1', 'DY2']
        print(f"未找到任何预处理文件，尝试默认受试者列表: {default_subjects}")
        subjects = default_subjects
    
    # 解析时间点
    if args.time_points:
        time_points = [int(t.strip()) for t in args.time_points.split(',')]
    else:
        time_points = None
    
    # 为每个受试者可视化数据
    for subject in subjects:
        visualize_subject_data(
            subject, 
            args.data_dir, 
            args.output_dir, 
            samples_per_class=args.samples,
            time_points=time_points,
            create_animations=args.animations
        )
    
    print(f"所有可视化任务完成！结果保存在 {args.output_dir}")

if __name__ == "__main__":
    main()