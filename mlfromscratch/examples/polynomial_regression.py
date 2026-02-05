from __future__ import print_function  # 向后兼容的 print 函数导入（针对旧版 Python）
import matplotlib.pyplot as plt  # 导入 matplotlib 的绘图子模块
import numpy as np  # 导入 numpy 用于数值计算
import pandas as pd  # 导入 pandas 用于读取和处理表格数据
# Import helper functions  # 导入一些辅助函数和模型类
from mlfromscratch.supervised_learning import PolynomialRidgeRegression  # 多项式岭回归模型类
from mlfromscratch.utils import k_fold_cross_validation_sets, normalize, mean_squared_error  # 导入交叉验证、归一化和 MSE
from mlfromscratch.utils import train_test_split, polynomial_features, Plot  # 导入训练/测试拆分、多项式特征和绘图工具


def main():  # 主函数入口

    # Load temperature data  # 载入温度数据文件
    data = pd.read_csv('mlfromscratch/data/TempLinkoping2016.txt', sep="\t")  # 使用制表符分隔的文本文件

    time = np.atleast_2d(data["time"].values).T  # 把时间列转换为列向量（2D 数组）
    temp = data["temp"].values  # 温度列作为一维数组

    X = time  # X 为一年中时间（0 到 1 之间的分数）
    y = temp  # y 为对应的温度

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)  # 按 60/40 划分训练/测试集

    poly_degree = 15  # 多项式的阶数（15 次）

    # Finding regularization constant using cross validation  # 使用交叉验证寻找最优正则化参数
    lowest_error = float("inf")  # 初始化最低误差为无穷大
    best_reg_factor = None  # 最佳正则化系数初始化为 None
    print ("Finding regularization constant using cross validation:")  # 打印提示信息
    k = 10  # 10 折交叉验证
    for reg_factor in np.arange(0, 0.1, 0.01):  # 在 0 到 0.09 之间以 0.01 步长尝试正则化系数
        cross_validation_sets = k_fold_cross_validation_sets(
            X_train, y_train, k=k)  # 生成 k 折交叉验证的数据集
        mse = 0  # 用于累加每折的均方误差
        for _X_train, _X_test, _y_train, _y_test in cross_validation_sets:  # 遍历每一折
            model = PolynomialRidgeRegression(degree=poly_degree, 
                                            reg_factor=reg_factor,
                                            learning_rate=0.001,
                                            n_iterations=10000)  # 初始化模型并设置超参数
            model.fit(_X_train, _y_train)  # 在当前折的训练集上训练模型
            y_pred = model.predict(_X_test)  # 对当前折的测试集做预测
            _mse = mean_squared_error(_y_test, y_pred)  # 计算均方误差
            mse += _mse  # 累加本折误差
        mse /= k  # 计算平均误差（取 k 折的平均）

        # Print the mean squared error  # 打印当前正则化系数下的平均 MSE
        print ("\tMean Squared Error: %s (regularization: %s)" % (mse, reg_factor))

        # Save reg. constant that gave lowest error  # 更新最优正则化系数
        if mse < lowest_error:
            best_reg_factor = reg_factor  # 记录最佳正则化系数
            lowest_error = mse  # 更新最低误差

    # Make final prediction  # 使用最佳正则化系数在全部训练集上训练最终模型并预测
    model = PolynomialRidgeRegression(degree=poly_degree, 
                                    reg_factor=best_reg_factor,
                                    learning_rate=0.001,
                                    n_iterations=10000)  # 使用找到的 best_reg_factor 初始化最终模型
    model.fit(X_train, y_train)  # 在训练集上训练最终模型
    y_pred = model.predict(X_test)  # 在测试集上做预测
    mse = mean_squared_error(y_test, y_pred)  # 计算最终模型的 MSE
    print ("Mean squared error: %s (given by reg. factor: %s)" % (lowest_error, best_reg_factor))  # 打印结果

    y_pred_line = model.predict(X)  # 在所有数据点上预测用于绘图的曲线

    # Color map  # 配色方案
    cmap = plt.get_cmap('viridis')  # 选择 viridis 颜色映射

    # Plot the results  # 绘制训练/测试点和预测曲线
    m1 = plt.scatter(366 * X_train, y_train, color=cmap(0.9), s=10)  # 绘制训练数据点（按天数放缩）
    m2 = plt.scatter(366 * X_test, y_test, color=cmap(0.5), s=10)  # 绘制测试数据点
    plt.plot(366 * X, y_pred_line, color='black', linewidth=2, label="Prediction")  # 绘制预测曲线
    plt.suptitle("Polynomial Ridge Regression")  # 主标题
    plt.title("MSE: %.2f" % mse, fontsize=10)  # 副标题显示 MSE
    plt.xlabel('Day')  # x 轴标签
    plt.ylabel('Temperature in Celcius')  # y 轴标签
    plt.legend((m1, m2), ("Training data", "Test data"), loc='lower right')  # 图例
    plt.show()  # 显示图形窗口


if __name__ == "__main__":
    main()  # 当作脚本直接运行时执行 main()
