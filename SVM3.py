import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import svm
from scipy.spatial.distance import cdist

# 定义拉普拉斯核函数
def laplacian_kernel(X, Y, gamma):
    return np.exp(-gamma * cdist(X, Y))

# 加载数据集
lung_data = pd.read_csv(r"datasets\lung.csv")

# 获取特征和标签
X = lung_data.iloc[:, :-1].values  # 特征矩阵
y = lung_data.iloc[:, -1].values   # 目标向量

# 分割训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 设置 gamma 的值
gamma = 1

# 计算训练数据的核矩阵
K_train = laplacian_kernel(X_train, X_train, gamma)

# 使用预计算的核矩阵训练 SVM 模型
svm_classifier = svm.SVC(kernel='precomputed')
svm_classifier.fit(K_train, y_train)

# 计算测试数据的核矩阵
K_test = laplacian_kernel(X_test, X_train, gamma)

# 在测试集上应用模型
predictions = svm_classifier.predict(K_test)
accuracy = np.mean(predictions == y_test)


# 打印预测结果及模型评分
print("Predicted labels: ",predictions)
print("Accuracy score: ", accuracy)



