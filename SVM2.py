# encoding:utf-8
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# 加载数据集
lung_data = pd.read_csv(r"datasets\Yale.csv")

# 获取特征和标签
X = lung_data.iloc[:, :-1]  # 特征矩阵
y = lung_data.iloc[:, -1]   # 目标向量

# 数据集切分
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)

# SVM分类器模型训练（使用高斯核）
svm_model = SVC(kernel='rbf', C=1.0)
svm_model.fit(X_train, y_train)

# 预测测试数据集
predicted_y = svm_model.predict(X_test)

# 计算准确率
accuracy = accuracy_score(y_test, predicted_y)

# 打印预测结果及模型评分
print("Predicted labels: ", predicted_y)
print("Accuracy score: ", accuracy)
