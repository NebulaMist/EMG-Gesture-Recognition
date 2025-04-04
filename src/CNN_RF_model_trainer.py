def create_cnn_model(input_shape, num_classes):
    """创建CNN模型"""
    model = models.Sequential()
    
    # 第一阶段：特征提取
    # 输入形状: (time_points, 16, 16, 1)
    model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=input_shape))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2, 2)))
    
    # 第二阶段：分类
    model.add(layers.Flatten())
    model.add(layers.Dropout(0.5))  # 防止过拟合
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dropout(0.5))  # 防止过拟合
    model.add(layers.Dense(num_classes, activation='softmax'))
    
    # 编译模型
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model
def train_and_evaluate_cnn(X_train, y_train, X_test, y_test, config, num_classes):
    """训练并评估CNN模型"""
    print("创建CNN模型...")
    
    # 调整输入形状为CNN需要的格式
    # 从 (n_samples, time_points, 16, 16) 转换为 (n_samples, 16, 16, time_points)
    X_train = np.transpose(X_train, (0, 2, 3, 1))
    X_test = np.transpose(X_test, (0, 2, 3, 1))
    
    # 如果时间点维度为1，则去掉该维度
    if X_train.shape[3] == 1:
        X_train = X_train.squeeze(axis=3)
        X_test = X_test.squeeze(axis=3)
        # 添加通道维度
        X_train = np.expand_dims(X_train, axis=-1)
        X_test = np.expand_dims(X_test, axis=-1)
        input_shape = (16, 16, 1)  # (height, width, channels)
    else:
        input_shape = (16, 16, X_train.shape[3])  # (height, width, time_points)
    
    # 创建模型
    model = create_cnn_model(input_shape, num_classes)
    model.summary()
    
    # 创建回调函数
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            patience=10, 
            restore_best_weights=True,
            monitor='val_accuracy'
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(config.save_dir, f"cnn_model_{config.timestamp}.h5"),
            save_best_only=True,
            monitor='val_accuracy'
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            factor=0.5, 
            patience=5,
            min_lr=0.00001,
            monitor='val_loss'
        )
    ]
    
    # 训练模型
    print("\n开始训练CNN模型...")
    history = model.fit(
        X_train, y_train,
        epochs=config.epochs,
        batch_size=config.batch_size,
        validation_split=config.validation_split,
        callbacks=callbacks
    )
    
    # 评估模型
    print("\n在测试集上评估CNN模型...")
    test_loss, test_acc = model.evaluate(X_test, y_test)
    print(f"测试集准确率: {test_acc:.4f}")
    
    return model, history, test_acc
def main():
    """主程序流程"""
    config = CNNConfig()
    
    # 1. 加载原始EMG数据
    raw_emg_data, labels, subject_ids = load_raw_emg_data(config)
    
    # 2. 预处理数据并重组为16x16矩阵
    processed_data, one_hot_labels, num_classes = preprocess_for_cnn(raw_emg_data, labels, config)
    print(f"预处理后的数据形状: {processed_data.shape}")
    print(f"One-hot标签形状: {one_hot_labels.shape}")
    print(f"类别数: {num_classes}")
    
    # 3. 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        processed_data, one_hot_labels,
        test_size=0.2,
        random_state=42,
        stratify=labels  # 确保类别分布一致
    )
    
    # 4. 训练和评估CNN模型
    model, history, test_acc = train_and_evaluate_cnn(X_train, y_train, X_test, y_test, config, num_classes)
    
    # 5. 绘制训练过程
    plt.figure(figsize=(12, 5))
    
    # 准确率
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('模型准确率')
    plt.ylabel('准确率')
    plt.xlabel('Epoch')
    plt.legend(['训练集', '验证集'], loc='lower right')
    
    # 损失
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('模型损失')
    plt.ylabel('损失')
    plt.xlabel('Epoch')
    plt.legend(['训练集', '验证集'], loc='upper right')
    
    # 保存图表
    plt.tight_layout()
    plt.savefig(os.path.join(config.save_dir, f"training_history_{config.timestamp}.png"))
    print(f"训练历史已保存至 {os.path.join(config.save_dir, f'training_history_{config.timestamp}.png')}")
    
    # 保存测试结果
    results = {
        "test_accuracy": test_acc,
        "timestamp": config.timestamp
    }
    np.save(os.path.join(config.save_dir, f"test_results_{config.timestamp}.npy"), results)
    print(f"测试结果已保存至 {os.path.join(config.save_dir, f'test_results_{config.timestamp}.npy')}")

if __name__ == "__main__":
    main()