import subprocess
import sys
import platform
import os

def install_tensorflow():
    """安装TensorFlow和相关依赖"""
    print("开始安装TensorFlow及相关库...")
    
    # 检查Python版本
    python_version = platform.python_version()
    print(f"Python版本: {python_version}")
    
    # 检查是否有GPU
    try:
        import subprocess
        gpu_info = subprocess.check_output('nvidia-smi', shell=True).decode('utf-8')
        print("发现NVIDIA GPU:")
        for line in gpu_info.split('\n')[:10]:  # 只显示前10行
            print(line)
        has_gpu = True
    except:
        print("未找到NVIDIA GPU或nvidia-smi无法执行")
        has_gpu = False
    
    # 安装必要的工具包
    print("\n安装/更新pip...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
    
    print("\n安装必要的依赖库...")
    packages = [
        "numpy",
        "scipy",
        "matplotlib",
        "scikit-learn",
        "h5py",
        "pillow",
        "pydot"
    ]
    
    for package in packages:
        print(f"安装 {package}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
    
    # 根据是否有GPU选择安装的TensorFlow版本
    if has_gpu:
        print("\n安装GPU版本的TensorFlow...")
        tensorflow_package = "tensorflow-gpu"
    else:
        print("\n安装CPU版本的TensorFlow...")
        tensorflow_package = "tensorflow"
    
    subprocess.check_call([sys.executable, "-m", "pip", "install", tensorflow_package])
    
    print("\n安装完成！")

def verify_tensorflow():
    """验证TensorFlow是否正确安装并能够使用"""
    try:
        import tensorflow as tf
        print(f"\nTensorFlow版本: {tf.__version__}")
        
        # 验证TensorFlow可以运行
        print("\n创建简单的TensorFlow操作进行测试...")
        hello = tf.constant('Hello, TensorFlow!')
        print(hello.numpy().decode('utf-8'))
        
        # 检查是否有GPU可用
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print(f"\n可用的GPU设备: {len(gpus)}")
            for gpu in gpus:
                print(f" - {gpu.name}")
            
            # 尝试限制内存增长
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                print("已启用GPU内存动态增长")
            except RuntimeError as e:
                print(f"设置内存增长失败: {e}")
        else:
            print("\n未检测到GPU设备，将使用CPU进行计算")
        
        # 简单的矩阵运算测试
        print("\n执行简单的矩阵运算测试...")
        a = tf.random.normal([1000, 1000])
        b = tf.random.normal([1000, 1000])
        c = tf.matmul(a, b)
        print(f"矩阵乘法结果形状: {c.shape}")
        
        print("\nTensorFlow验证成功！")
        return True
    except ImportError:
        print("\nTensorFlow导入失败，安装可能有问题")
        return False
    except Exception as e:
        print(f"\nTensorFlow验证过程中出错: {e}")
        return False

def check_environment():
    """检查系统环境"""
    print("系统信息:")
    print(f" - 操作系统: {platform.system()} {platform.version()}")
    print(f" - Python版本: {platform.python_version()}")
    print(f" - Python路径: {sys.executable}")
    
    # 检查CUDA和cuDNN
    if platform.system() == "Windows":
        # 尝试检查CUDA_PATH环境变量
        cuda_path = os.environ.get("CUDA_PATH", "未设置")
        print(f" - CUDA路径: {cuda_path}")
        
        # 尝试运行nvcc查看CUDA版本
        try:
            nvcc_output = subprocess.check_output("nvcc --version", shell=True).decode('utf-8')
            for line in nvcc_output.split('\n'):
                if "release" in line:
                    print(f" - CUDA版本: {line.strip()}")
                    break
        except:
            print(" - 无法确定CUDA版本")
    else:
        # Linux系统
        try:
            cuda_version = subprocess.check_output("cat /usr/local/cuda/version.txt 2>/dev/null || echo 'CUDA未安装'", shell=True).decode('utf-8')
            print(f" - CUDA版本: {cuda_version.strip()}")
        except:
            print(" - 无法确定CUDA版本")

if __name__ == "__main__":
    print("======== TensorFlow设置和验证工具 ========\n")
    
    check_environment()
    
    action = input("\n请选择操作:\n1. 安装TensorFlow\n2. 验证TensorFlow\n3. 全部执行\n选择 (默认:3): ") or "3"
    
    if action in ["1", "3"]:
        install_tensorflow()
    
    if action in ["2", "3"]:
        verify_tensorflow()
    
    print("\n======== 完成 ========")