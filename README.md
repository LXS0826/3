# Lung Cancer Classification using SVM with Laplacian Kernel

这个项目展示了如何使用拉普拉斯核函数的支持向量机（SVM）对肺部数据进行分类。我们自定义了一个拉普拉斯核函数，并使用 scikit-learn 的 SVM 分类器进行模型训练和预测。

## 安装指南

要运行这个项目，你需要安装以下 Python 包：

- pandas
- numpy
- scikit-learn
- scipy

你可以通过以下命令来安装这些包：

```bash
pip install pandas numpy scikit-learn scipy
数据集
我们使用了一个包含lung，minst，yale数据的 CSV 文件作为数据集。这个数据集包含了多个特征，以及一个目标标签，用于表示样本是否属于某个特定的肺部疾病类别。

确保你的数据文件 lung.csv 位于 datasets 文件夹中，以便脚本能够正确加载数据。

使用方法
克隆这个仓库到你的本地机器上：

git clone https://github.com/your-username/lung-cancer-svm-classification.git
cd lung-cancer-svm-classification
运行 script.py 文件：

python script.py
核函数
项目中定义了一个线性支持向量机（SVM），拉普拉斯核函数，高斯核，用于 SVM 分类器。
