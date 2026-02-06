from __future__ import print_function  # 向后兼容的 print 函数（兼容 Python2 的 print 语法）
from sklearn import datasets  # 导入 scikit-learn 的数据集模块
import numpy as np  # 导入 numpy 用于数值计算和数组操作
import math  # 导入 math 模块（如需数学函数）
import matplotlib.pyplot as plt  # 导入 matplotlib 的绘图模块

from mlfromscratch.utils import train_test_split, normalize, to_categorical, accuracy_score  # 导入工具函数：划分数据、归一化、独热编码、准确率
from mlfromscratch.deep_learning.optimizers import Adam  # 导入 Adam 优化器实现
from mlfromscratch.deep_learning.loss_functions import CrossEntropy  # 导入交叉熵损失函数
from mlfromscratch.deep_learning.activation_functions import Softmax  # 导入 Softmax 激活（若需）
from mlfromscratch.utils.kernels import *  # 导入核函数集合（在某些算法中使用）
from mlfromscratch.supervised_learning import *  # 导入仓库中的监督学习算法实现（如 KNN、决策树等）
from mlfromscratch.deep_learning import *  # 导入深度学习模块中的类和方法
from mlfromscratch.unsupervised_learning import PCA  # 导入 PCA 用于降维
from mlfromscratch.deep_learning.layers import Dense, Dropout, Conv2D, Flatten, Activation  # 导入常用层类型


print ("+-------------------------------------------+")  # 打印框状标题行
print ("|                                           |")  # 打印标题空行
print ("|       Machine Learning From Scratch       |")  # 打印项目名标题
print ("|                                           |")  # 打印标题空行
print ("+-------------------------------------------+")  # 打印框状标题行结束


# ...........
#  LOAD DATA
# ...........
data = datasets.load_digits()  # 加载 sklearn 的 digits 数据集（8x8 图片）
digit1 = 1  # 需要区分的第一个数字类别
digit2 = 8  # 需要区分的第二个数字类别
idx = np.append(np.where(data.target == digit1)[0], np.where(data.target == digit2)[0])  # 找到两个类别的样本索引并合并
y = data.target[idx]  # 根据索引获取对应的标签
# Change labels to {0, 1}
y[y == digit1] = 0  # 把 digit1 的标签改为 0
y[y == digit2] = 1  # 把 digit2 的标签改为 1
X = data.data[idx]  # 取相应索引的特征向量（每张图片为 64 维向量）
X = normalize(X)  # 对特征进行归一化处理

print ("Dataset: The Digit Dataset (digits %s and %s)" % (digit1, digit2))  # 打印当前使用的数据集信息

# ..........................
#  DIMENSIONALITY REDUCTION
# ..........................
pca = PCA()  # 实例化 PCA 对象
X = pca.transform(X, n_components=5) # Reduce to 5 dimensions  将数据降到 5 维

n_samples, n_features = np.shape(X)  # 获取样本数和特征维度

# ..........................
#  TRAIN / TEST SPLIT
# ..........................
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)  # 将数据按 50% 测试集划分训练/测试
# Rescaled labels {-1, 1}
rescaled_y_train = 2*y_train - np.ones(np.shape(y_train))  # 把标签从 {0,1} 变换到 {-1,1}（用于某些算法）
rescaled_y_test = 2*y_test - np.ones(np.shape(y_test))  # 同上，用于测试集

# .......
#  SETUP
# .......
adaboost = Adaboost(n_clf = 8)  # 实例化 Adaboost 分类器，弱分类器数量 8
naive_bayes = NaiveBayes()  # 实例化朴素贝叶斯分类器
knn = KNN(k=4)  # 实例化 KNN 分类器，k=4
logistic_regression = LogisticRegression()  # 实例化逻辑回归
mlp = NeuralNetwork(optimizer=Adam(), 
                    loss=CrossEntropy)  # 实例化多层感知器，指定优化器和损失
mlp.add(Dense(input_shape=(n_features,), n_units=64))  # 添加全连接层，输入维度为特征数，隐藏单元 64
mlp.add(Activation('relu'))  # 添加 ReLU 激活
mlp.add(Dense(n_units=64))  # 添加另一个全连接层，64 个单元
mlp.add(Activation('relu'))  # 添加 ReLU 激活
mlp.add(Dense(n_units=2))   # 添加输出层，输出单元数为 2（两个类别）
mlp.add(Activation('softmax'))  # 添加 softmax 激活以输出概率分布
perceptron = Perceptron()  # 实例化感知器分类器
decision_tree = ClassificationTree()  # 实例化分类决策树
random_forest = RandomForest(n_estimators=50)  # 实例化随机森林，树的数量为 50
support_vector_machine = SupportVectorMachine()  # 实例化支持向量机
lda = LDA()  # 实例化线性判别分析
gbc = GradientBoostingClassifier(n_estimators=50, learning_rate=.9, max_depth=2)  # 实例化梯度提升分类器
xgboost = XGBoost(n_estimators=50, learning_rate=0.5)  # 实例化 XGBoost

# ........
#  TRAIN
# ........
print ("Training:")  # 打印训练开始信息
print ("- Adaboost")  # 打印当前训练的模型名
adaboost.fit(X_train, rescaled_y_train)  # 训练 Adaboost（使用 {-1,1} 标签）
print ("- Decision Tree")  # 打印当前训练的模型名
decision_tree.fit(X_train, y_train)  # 训练决策树
print ("- Gradient Boosting")  # 打印当前训练的模型名
gbc.fit(X_train, y_train)  # 训练梯度提升分类器
print ("- LDA")  # 打印当前训练的模型名
lda.fit(X_train, y_train)  # 训练 LDA
print ("- Logistic Regression")  # 打印当前训练的模型名
logistic_regression.fit(X_train, y_train)  # 训练逻辑回归
print ("- Multilayer Perceptron")  # 打印当前训练的模型名
mlp.fit(X_train, to_categorical(y_train), n_epochs=300, batch_size=50)  # 训练 MLP，标签使用独热编码，迭代 300 次
print ("- Naive Bayes")  # 打印当前训练的模型名
naive_bayes.fit(X_train, y_train)  # 训练朴素贝叶斯
print ("- Perceptron")  # 打印当前训练的模型名
perceptron.fit(X_train, to_categorical(y_train))  # 训练感知器（使用独热编码标签）
print ("- Random Forest")  # 打印当前训练的模型名
random_forest.fit(X_train, y_train)  # 训练随机森林
print ("- Support Vector Machine")  # 打印当前训练的模型名
support_vector_machine.fit(X_train, rescaled_y_train)  # 训练 SVM（使用 {-1,1} 标签）
print ("- XGBoost")  # 打印当前训练的模型名
xgboost.fit(X_train, y_train)  # 训练 XGBoost



# .........
#  PREDICT
# .........
y_pred = {}  # 用于保存各模型的预测结果的字典
y_pred["Adaboost"] = adaboost.predict(X_test)  # Adaboost 的预测（输出 {-1,1}）
y_pred["Gradient Boosting"] = gbc.predict(X_test)  # GBC 的预测
y_pred["Naive Bayes"] = naive_bayes.predict(X_test)  # 朴素贝叶斯的预测
y_pred["K Nearest Neighbors"] = knn.predict(X_test, X_train, y_train)  # KNN 的预测（需要训练集作为参数）
y_pred["Logistic Regression"] = logistic_regression.predict(X_test)  # 逻辑回归的预测
y_pred["LDA"] = lda.predict(X_test)  # LDA 的预测
y_pred["Multilayer Perceptron"] = np.argmax(mlp.predict(X_test), axis=1)  # MLP 的预测（取概率最大索引）
y_pred["Perceptron"] = np.argmax(perceptron.predict(X_test), axis=1)  # 感知器的预测（取概率最大索引）
y_pred["Decision Tree"] = decision_tree.predict(X_test)  # 决策树的预测
y_pred["Random Forest"] = random_forest.predict(X_test)  # 随机森林的预测
y_pred["Support Vector Machine"] = support_vector_machine.predict(X_test)  # SVM 的预测（输出 {-1,1}）
y_pred["XGBoost"] = xgboost.predict(X_test)  # XGBoost 的预测

# ..........
#  ACCURACY
# ..........
print ("Accuracy:")  # 打印准确率前缀
for clf in y_pred:  # 遍历每个分类器
    # Rescaled {-1 1}
    if clf == "Adaboost" or clf == "Support Vector Machine":  # 对于输出为 {-1,1} 的模型
        print ("\t%-23s: %.5f" %(clf, accuracy_score(rescaled_y_test, y_pred[clf])))  # 计算并打印准确率
    # Categorical
    else:
        print ("\t%-23s: %.5f" %(clf, accuracy_score(y_test, y_pred[clf])))  # 对于常规分类器，用原始标签计算准确率

# .......
#  PLOT
# .......
plt.scatter(X_test[:,0], X_test[:,1], c=y_test)  # 在降维后的二维空间绘制测试样本，并根据真实标签着色
plt.ylabel("Principal Component 2")  # y 轴标签为第二主成分
plt.xlabel("Principal Component 1")  # x 轴标签为第一主成分
plt.title("The Digit Dataset (digits %s and %s)" % (digit1, digit2))  # 绘制图表标题
plt.show()  # 显示绘图窗口


