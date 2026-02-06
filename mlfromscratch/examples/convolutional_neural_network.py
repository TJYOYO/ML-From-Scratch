# -*- coding: utf-8 -*-  # 指定文件编码为 UTF-8
from __future__ import print_function  # 向后兼容的 print 函数导入（针对旧版 Python）
from sklearn import datasets  # 从 scikit-learn 导入数据集加载工具
import matplotlib.pyplot as plt  # 导入 matplotlib 的绘图子模块
import math  # 导入数学函数库（如需数学运算）
import numpy as np  # 导入 numpy 用于数组和数值计算

# Import helper functions  # 导入深度学习和工具模块
from mlfromscratch.deep_learning import NeuralNetwork  # 导入自定义的神经网络类
from mlfromscratch.utils import train_test_split, to_categorical, normalize  # 导入数据处理工具
from mlfromscratch.utils import get_random_subsets, shuffle_data, Plot  # 导入额外的工具函数和绘图助手
from mlfromscratch.utils.data_operation import accuracy_score  # 导入准确率评估函数
from mlfromscratch.deep_learning.optimizers import StochasticGradientDescent, Adam, RMSprop, Adagrad, Adadelta  # 导入优化器实现
from mlfromscratch.deep_learning.loss_functions import CrossEntropy  # 导入交叉熵损失函数
from mlfromscratch.utils.misc import bar_widgets  # 导入命令行进度条工具（如需）
from mlfromscratch.deep_learning.layers import Dense, Dropout, Conv2D, Flatten, Activation, MaxPooling2D  # 导入常用层
from mlfromscratch.deep_learning.layers import AveragePooling2D, ZeroPadding2D, BatchNormalization, RNN  # 导入其它层和循环层


def main():  # 主程序入口函数

    #----------
    # Conv Net
    #----------

    optimizer = Adam()  # 使用 Adam 优化器实例化优化器对象

    data = datasets.load_digits()  # 加载 scikit-learn 手写数字数据集
    X = data.data  # 特征矩阵，形状 (n_samples, n_features)
    y = data.target  # 标签向量

    # Convert to one-hot encoding  # 将标签转换为 one-hot 编码
    y = to_categorical(y.astype("int"))  # 强制为整型后做独热编码

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, seed=1)  # 划分训练/测试集，测试集占 40%

    # Reshape X to (n_samples, channels, height, width)  # 将平展的图片数据重塑为 4D 张量
    X_train = X_train.reshape((-1,1,8,8))  # 训练集重塑为 (样本数, 1, 8, 8)
    X_test = X_test.reshape((-1,1,8,8))  # 测试集重塑为 (样本数, 1, 8, 8)

    clf = NeuralNetwork(optimizer=optimizer,
                        loss=CrossEntropy,
                        validation_data=(X_test, y_test))  # 实例化神经网络，指定优化器、损失和验证数据

    clf.add(Conv2D(n_filters=16, filter_shape=(3,3), stride=1, input_shape=(1,8,8), padding='same'))  # 添加卷积层（16 个过滤器，3x3，same 填充）
    clf.add(Activation('relu'))  # 添加 ReLU 激活层
    clf.add(Dropout(0.25))  # 添加 dropout 层以减缓过拟合，丢弃率 25%
    clf.add(BatchNormalization())  # 添加批归一化层
    clf.add(Conv2D(n_filters=32, filter_shape=(3,3), stride=1, padding='same'))  # 再添加一层卷积层（32 个过滤器）
    clf.add(Activation('relu'))  # 添加 ReLU 激活
    clf.add(Dropout(0.25))  # 添加 dropout
    clf.add(BatchNormalization())  # 添加批归一化
    clf.add(Flatten())  # 将多维特征展平为一维向量以输入全连接层
    clf.add(Dense(256))  # 添加全连接层，256 个神经元
    clf.add(Activation('relu'))  # 添加 ReLU 激活
    clf.add(Dropout(0.4))  # 添加 dropout，丢弃率 40%
    clf.add(BatchNormalization())  # 添加批归一化
    clf.add(Dense(10))  # 添加输出层，10 个神经元（对应 10 个类别）
    clf.add(Activation('softmax'))  # 添加 softmax 激活以输出概率分布

    print ()  # 打印空行
    clf.summary(name="ConvNet")  # 打印模型摘要，显示网络结构和参数

    train_err, val_err = clf.fit(X_train, y_train, n_epochs=50, batch_size=256)  # 训练模型，返回训练和验证误差序列

    # Training and validation error plot  # 绘制训练误差与验证误差曲线
    n = len(train_err)  # 获取训练误差序列长度（迭代次数或 epoch 数）
    training, = plt.plot(range(n), train_err, label="Training Error")  # 绘制训练误差曲线
    validation, = plt.plot(range(n), val_err, label="Validation Error")  # 绘制验证误差曲线
    plt.legend(handles=[training, validation])  # 显示图例
    plt.title("Error Plot")  # 图表标题
    plt.ylabel('Error')  # y 轴标签
    plt.xlabel('Iterations')  # x 轴标签
    plt.show()  # 显示图形窗口

    _, accuracy = clf.test_on_batch(X_test, y_test)  # 在测试集上评估模型并得到准确率
    print ("Accuracy:", accuracy)  # 打印准确率


    y_pred = np.argmax(clf.predict(X_test), axis=1)  # 获取测试集上预测的类别（取最大概率索引）
    X_test = X_test.reshape(-1, 8*8)  # 将测试集重塑回平展格式以便降维可视化
    # Reduce dimension to 2D using PCA and plot the results  # 使用 PCA 降至 2D 并绘图
    Plot().plot_in_2d(X_test, y_pred, title="Convolutional Neural Network", accuracy=accuracy, legend_labels=range(10))  # 绘制 2D 可视化

if __name__ == "__main__":
    main()  # 如果作为脚本运行则执行 main()
