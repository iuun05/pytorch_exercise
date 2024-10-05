import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 为了和自己计算的数据进行对比
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


class dataprocess:
    def __init__(self, data_path):
        self.threshold = 130000
        self.data = pd.read_csv(data_path).values
        self.GPA = np.array(self.data[:, 0])

        self.Internships = np.array(self.data[:, 1])
        self.Competitions = np.array(self.data[:, 2])
        self.Penalties = np.array(self.data[:, 3])
        self.Nationality = np.array(self.data[:, 4])
        self.Salary = np.array(self.data[:, 5])
        # 将工资大于阈值的标记为1，小于的标记为0
        self.Salary_Classification = np.where(self.Salary > self.threshold, 1, 0)
        self.C = np.array([self.data[100 * i:100 * (i + 1), :] for i in range(5)])

        self.Nationality_onehot = self.one_hot_encoder(self.Nationality)

        self.batchs = [1, 4, 8, 16]

        self.weights = None
        self.feature = None

        self.beta = None  # 存储训练后的权重

    def Visualization(self):
        fig = plt.figure(figsize=(16, 12))
        ax1 = fig.add_subplot(221)
        ax1.hist(self.Salary, bins=15, color='#f59311', alpha=0.5, edgecolor='k')
        ax1.set_title('Salary')
        ax1.set_xlabel('Salary')
        ax1.set_ylabel('Frequency')

        ax2 = fig.add_subplot(222)
        ax2.scatter(self.GPA, self.Salary, color='#f59311', alpha=0.5, edgecolor='k')
        ax2.set_title('GPA vs Salary')
        ax2.set_xlabel('GPA')
        ax2.set_ylabel('Salary')

        ax3 = fig.add_subplot(223)
        ax3.scatter(self.Internships, self.Salary, color='#f59311', alpha=0.5, edgecolor='k')
        ax3.set_title('Internships vs Salary')
        ax3.set_xlabel('Internships')
        ax3.set_ylabel('Salary')

        ax4 = fig.add_subplot(224)
        ax4.scatter(self.Competitions, self.Salary, color='#f59311', alpha=0.5, edgecolor='k')
        ax4.set_title('Competitions vs Salary')
        ax4.set_xlabel('Competitions')
        ax4.set_ylabel('Salary')

        plt.show()

        fig = plt.figure(figsize=(16, 12))
        ax5 = fig.add_subplot(231)
        ax5.scatter(self.Penalties, self.Salary, color='#f59311', alpha=0.5, edgecolor='k')
        ax5.set_title('Penalties vs Salary')
        ax5.set_xlabel('Penalties')
        ax5.set_ylabel('Salary')

        ax6 = fig.add_subplot(232)
        ax6.scatter(self.Nationality, self.Salary, color='#f59311', alpha=0.5, edgecolor='k')
        ax6.set_title('Nationality vs Salary')
        ax6.set_xlabel('Nationality')
        ax6.set_ylabel('Salary')

        plt.show()

    def average(self, C, K=None):
        # data = np.array([])
        # for i in range(5):
        #     if i is not K:
        #         data = np.hstack((data, C[100*i:100*(i+1)]))
        return np.mean(C, axis=0)

    def variance(self, C, K=None):
        # data = np.array([])
        # for i in range(5):
        #     if i is not K:
        #         data = np.hstack((data, C[100 * i:100 * (i + 1)]))

        return np.var(C, axis=0)

    def one_hot_encoder(self, Nationality):

        unique_categories = np.unique(Nationality)

        category_to_index = {category: index for index, category in enumerate(unique_categories)}

        # 初始化 One-Hot 编码矩阵
        Nationality_onehot = np.zeros((len(Nationality), len(unique_categories)))

        # 将类别映射为 One-Hot 向量
        for i, category in enumerate(Nationality):
            Nationality_onehot[i, category_to_index[category]] = 1

        # print("Unique categories:", unique_categories)
        # print("One-Hot Encoded matrix:")
        # print(self.Nationality_onehot)

        return Nationality_onehot

    def standardization(self, C, average, variance, k):
        return (C - average) / variance

    def Least_Square(self, train_data, train_label, test_data, test_label):
        X_transpose = np.transpose(train_data)

        XTX = np.dot(X_transpose, train_data)

        XTX_inv = np.linalg.inv(XTX)

        XTy = np.dot(X_transpose, train_label)

        self.beta = np.dot(XTX_inv, XTy)

        train_predictions = np.dot(train_data, self.beta)
        test_predictions = np.dot(test_data, self.beta)

        train_mse = np.mean((train_predictions - train_label) ** 2)
        test_mse = np.mean((test_predictions - test_label) ** 2)

        # 输出结果
        # print("Model weights (beta):", self.beta)
        # print("Training MSE:", train_mse)
        # print("Test MSE:", test_mse)

        return self.beta, train_mse, test_mse

    def task3_dataprocess(self, i):
        # 数据归一化，以及相关的划分
        GPA_average = self.average(np.delete(self.GPA, slice(100 * i, 100 * (i + 1))), i)
        GPA_variance = self.variance(np.delete(self.GPA, slice(100 * i, 100 * (i + 1))), i)
        GPA = self.standardization(self.GPA, GPA_average, GPA_variance, i)

        GPA_test = GPA[100 * i:100 * (i + 1)]
        GPA = np.delete(GPA, slice(100 * i, 100 * (i + 1)), axis=0)

        Interships_test = self.Internships[100 * i:100 * (i + 1)]
        Interships = np.delete(self.Internships, slice(100 * i, 100 * (i + 1)), axis=0)

        Competitions_test = self.Competitions[100 * i:100 * (i + 1)]
        Competitions = np.delete(self.Competitions, slice(100 * i, 100 * (i + 1)), axis=0)

        Penalties_test = self.Penalties[100 * i:100 * (i + 1)]
        Penalties = np.delete(self.Penalties, slice(100 * i, 100 * (i + 1)), axis=0)

        Nationality_onehot = self.Nationality_onehot
        Nationality_test = Nationality_onehot[100 * i:100 * (i + 1)]
        Nationality = np.delete(Nationality_onehot, slice(100 * i, 100 * (i + 1)), axis=0)

        # Salary_average = self.average(np.delete(self.Salary, slice(100 * i, 100 * (i + 1))), i)
        # Salary_variance = self.variance(np.delete(self.Salary, slice(100 * i, 100 * (i + 1))), i)
        # Salary = self.standardization(self.Salary, Salary_average, Salary_variance, i)
        # high = 1, low = 0
        Salary = self.Salary_Classification
        # print(Salary)
        # print(Salary.shape)
        Salary_test = Salary[100 * i:100 * (i + 1)]
        Salary = np.delete(Salary, slice(100 * i, 100 * (i + 1)), axis=0)

        # train_data = np.hstack(GPA, Interships, Competitions, Penalties, Nationality)
        train_data = np.concatenate((GPA[:, np.newaxis], Interships[:, np.newaxis], Competitions[:, np.newaxis],
                                     Penalties[:, np.newaxis], Nationality), axis=1).astype(float)

        train_label = Salary.astype(float)

        test_data = np.concatenate(
            (GPA_test[:, np.newaxis], Interships_test[:, np.newaxis], Competitions_test[:, np.newaxis],
             Penalties_test[:, np.newaxis], Nationality_test), axis=1).astype(float)
        test_label = Salary_test.astype(float)

        return train_data, train_label, test_data, test_label



    def pre_dataprocess(self, i):
        # 数据归一化，以及相关的划分
        GPA_average = self.average(np.delete(self.GPA, slice(100 * i, 100 * (i + 1))), i)
        GPA_variance = self.variance(np.delete(self.GPA, slice(100 * i, 100 * (i + 1))), i)
        GPA = self.standardization(self.GPA, GPA_average, GPA_variance, i)

        GPA_test = GPA[100 * i:100 * (i + 1)]
        GPA = np.delete(GPA, slice(100 * i, 100 * (i + 1)), axis=0)

        Interships_test = self.Internships[100 * i:100 * (i + 1)]
        Interships = np.delete(self.Internships, slice(100 * i, 100 * (i + 1)), axis=0)

        Competitions_test = self.Competitions[100 * i:100 * (i + 1)]
        Competitions = np.delete(self.Competitions, slice(100 * i, 100 * (i + 1)), axis=0)

        Penalties_test = self.Penalties[100 * i:100 * (i + 1)]
        Penalties = np.delete(self.Penalties, slice(100 * i, 100 * (i + 1)), axis=0)

        Nationality_onehot = self.Nationality_onehot
        Nationality_test = Nationality_onehot[100 * i:100 * (i + 1)]
        Nationality = np.delete(Nationality_onehot, slice(100 * i, 100 * (i + 1)), axis=0)

        Salary_average = self.average(np.delete(self.Salary, slice(100 * i, 100 * (i + 1))), i)
        Salary_variance = self.variance(np.delete(self.Salary, slice(100 * i, 100 * (i + 1))), i)
        Salary = self.standardization(self.Salary, Salary_average, Salary_variance, i)

        Salary_test = Salary[100 * i:100 * (i + 1)]
        Salary = np.delete(Salary, slice(100 * i, 100 * (i + 1)), axis=0)

        # train_data = np.hstack(GPA, Interships, Competitions, Penalties, Nationality)
        train_data = np.concatenate((GPA[:, np.newaxis], Interships[:, np.newaxis], Competitions[:, np.newaxis],
                                     Penalties[:, np.newaxis], Nationality), axis=1).astype(float)

        train_label = Salary.astype(float)

        test_data = np.concatenate(
            (GPA_test[:, np.newaxis], Interships_test[:, np.newaxis], Competitions_test[:, np.newaxis],
             Penalties_test[:, np.newaxis], Nationality_test), axis=1).astype(float)
        test_label = Salary_test.astype(float)

        return train_data, train_label, test_data, test_label


    def K_fold(self, k=5):
        # GPA, Internships, Competitions, Penalties, Nationality, Salary
        # 最小二乘法的储存
        beta_array, train_mse_array, test_mse_array = [], [], []
        # SGD的数据结构
        # weight_dict, bias_dict, train_loss_dict, test_loss_dict = {}, {}, {}, {}
        for i in range(k):
            train_data, train_label, test_data, test_label = self.pre_dataprocess(i)
            # 最小二乘法
            beta, train_mse, test_mse = self.Least_Square(train_data, train_label, test_data, test_label)

            beta_array.append(beta)
            train_mse_array.append(train_mse)
            test_mse_array.append(test_mse)


        print("############################################")
        print('my K-fold function:')
        print(f"beta: {np.mean(beta_array)} + {np.var(beta_array)}")
        print(f"train_mse: {np.mean(train_mse_array)} + {np.var(train_mse_array)}")
        print(f"test_mse: {np.mean(test_mse_array)} + {np.var(test_mse_array)}")

        x_labels = [f'Run {i + 1}' for i in range(5)]
        x = np.arange(len(x_labels))  # X 轴的位置
        width = 0.35  # 柱子的宽度

        fig, ax = plt.subplots()
        # 绘制柱状图

        train_mse_array_scaled = [value * 1e15 for value in train_mse_array]
        test_mse_array_scaled = [value * 1e15 for value in test_mse_array]

        # 绘制训练 MSE 和测试 MSE 的柱状图
        rects1 = ax.bar(x - width / 2, train_mse_array_scaled, width, label='Training MSE', color='b')
        rects2 = ax.bar(x + width / 2, test_mse_array_scaled, width, label='Testing MSE', color='r')

        ax.set_ylabel('MSE / e^-15')
        ax.set_title('Training and Testing MSE for Each Run')
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels)
        ax.legend()

        for rect in rects1:
            height = rect.get_height()
            ax.annotate(f'{height:.2f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 偏移量
                        textcoords="offset points",
                        ha='center', va='bottom')

        for rect in rects2:
            height = rect.get_height()
            ax.annotate(f'{height:.2f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 偏移量
                        textcoords="offset points",
                        ha='center', va='bottom')

        plt.tight_layout()
        plt.show()

        ####################################
        # sklearn库进行验证
        ####################################
        kf = KFold(n_splits=5)
        model = LinearRegression()
        train_mse_list, test_mse_list = [], []
        for train_index, test_index in kf.split(train_data):
            train_data_kf, test_data_kf = train_data[train_index], train_data[test_index]
            train_label_kf, test_label_kf = train_label[train_index], train_label[test_index]

            model.fit(train_data_kf, train_label_kf)

            train_pred = model.predict(train_data_kf)
            test_pred = model.predict(test_data_kf)

            train_mse = mean_squared_error(train_label_kf, train_pred)
            test_mse = mean_squared_error(test_label_kf, test_pred)

            train_mse_list.append(train_mse)
            test_mse_list.append(test_mse)

        print("############################################")
        print('sklearn K-fold function:')
        print("beta: ", model.intercept_)
        print("train_mse: ", np.mean(train_mse_list))
        print("test_mse: ", np.mean(test_mse_list))
        print("############################################")

    def SGD(self, train_data, train_label, test_data, test_label, learning_rate=0.001, max_iter=100, batch=1):

        n_samples, n_features = train_data.shape
        weights = np.zeros(n_features)
        bias = 0
        train_loss_list, test_loss_list = [], []
        for epoch in range(max_iter):
            for i in range(0, n_samples, batch):
                X_batch = train_data[i:i + batch]
                y_batch = train_label[i:i + batch]

                y_pred = np.dot(X_batch, weights) + bias

                gradient_w = -2 * np.dot(X_batch.T, (y_batch - y_pred)) / batch
                gradient_b = -2 * np.mean(y_batch - y_pred)

                weights -= learning_rate * gradient_w
                bias -= learning_rate * gradient_b

            train_loss = np.mean((train_label - (np.dot(train_data, weights) + bias)) ** 2) * 1e11
            train_loss_list.append(train_loss)

        test_loss = np.mean((test_label - (np.dot(test_data, weights) + bias)) ** 2) * 1e11
        test_loss_list.append(test_loss)

            # if epoch % 10 == 0:
            #     print(f"Epoch {epoch + 1}/{max_iter}, Training Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}")

        return weights, bias, train_loss_list, test_loss_list

    def mysgd(self, k=5):
        # GPA, Internships, Competitions, Penalties, Nationality, Salary
        # SGD的数据结构
        batches = [1, 4, 8, 16]
        weight_dict, bias_dict, train_loss_dict, test_loss_dict = {}, {}, {}, {}

            # SGD算法
        for batch in batches:
            weight_b = []
            bias_b = []
            train_loss_b = []
            test_loss_b = []
            for i in range(k):
                train_data, train_label, test_data, test_label = self.pre_dataprocess(i)

                sgd_weight, sgd_bias, train_loss_mse, test_loss_mse = self.SGD(train_data, train_label, test_data, test_label, learning_rate=0.001, batch=batch)

                weight_b.append(sgd_weight)
                bias_b.append(sgd_bias)
                train_loss_b.append(train_loss_mse)
                test_loss_b.append(test_loss_mse)

            weight_dict[batch] = np.mean(weight_b, axis=0)
            bias_dict[batch] = np.mean(bias_b, axis=0)
            train_loss_dict[batch] = np.mean(train_loss_b, axis=0)
            test_loss_dict[batch] = np.mean(test_loss_b, axis=0)

        fig = plt.figure(figsize=(16, 12))
        print("############################################")
        print('my SGD function:')
        for batch in batches:
            print("############################################")
            print("batch: ", batch)
            print("weight: ", weight_dict[batch])
            print("bias: ", bias_dict[batch])
            # print("train_mse: ", train_loss_dict[batch])
            # print("test_mse: ", test_loss_dict[batch])
            ax = fig.add_subplot(2, 2, batches.index(batch) + 1)
            ax.plot(train_loss_dict[batch], label='train_mse')
            ax.plot(test_loss_dict[batch], label='test_mse')
            ax.set_title(f'Batch Size: {batch}')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('MSE ^ 1e-11')
            ax.legend()

        print("############################################")
        plt.show()

    def task1(self):
        self.Visualization()
        self.K_fold()

    def task2(self):
        self.mysgd()


class Perceptron:
    def __init__(self, learning_rate=0.001, max_iter=100):
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.weights = None
        self.bias = None
        self.dp = dataprocess('salary.csv')

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    def SGD(self, train_data, train_label, test_data, test_label, learning_rate=0.001, max_iter=100, batch=1):

        n_samples, n_features = train_data.shape
        weights = np.zeros(n_features)
        bias = 0
        train_loss_list, test_loss_list = [], []

        for epoch in range(max_iter):

            for i in range(0, n_samples, batch):
                X_batch = train_data[i:i + batch]
                y_batch = train_label[i:i + batch]

                y_pred = self.sigmoid(np.dot(X_batch, weights) + bias)

                gradient_w = -2 * np.dot(X_batch.T, (y_batch - y_pred)) / batch
                gradient_b = -2 * np.mean(y_batch - y_pred)

                weights -= learning_rate * gradient_w
                bias -= learning_rate * gradient_b

                y_train_pred = self.sigmoid(np.dot(train_data, weights) + bias)
                train_loss = -np.mean(
                    train_label * np.log(y_train_pred + 1e-9) + (1 - train_label) * np.log(1 - y_train_pred + 1e-9))
                train_loss_list.append(train_loss)

            y_test_pred = self.sigmoid(np.dot(test_data, weights) + bias)
            test_loss = -np.mean(test_label * np.log(y_test_pred + 1e-9) + (1 - test_label) * np.log(1 - y_test_pred + 1e-9))
            test_loss_list.append(test_loss)

            # if epoch % 10 == 0:
            #     print(f"Epoch {epoch + 1}/{max_iter}, Training Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}")

        return weights, bias, train_loss_list, test_loss_list

    def train(self, k=5):
        weight_list, bias_list,train_mse_list, test_mse_list = [], [], [], []
        for i in range(k):
            train_data, train_label, test_data, test_label = self.dp.task3_dataprocess(i)
            weight, bias, train_mse_error, test_mse_error = self.SGD(train_data, train_label, test_data, test_label, learning_rate=0.001, batch=1)

            weight_list.append(weight)
            bias_list.append(bias)
            train_mse_list.append(train_mse_error)
            test_mse_list.append(test_mse_error)

        print("############################################")
        print('Perceptron function:')
        print("############################################")
        print("weight: ", np.mean(weight_list, axis=0))
        print("bias: ", np.mean(bias_list, axis=0))
        print("train_mse: ", np.mean(train_mse_list, axis=0))
        print("test_mse: ", np.mean(test_mse_list, axis=0))

        plt.plot(np.mean(train_mse_list, axis=1), label='train_mse')
        plt.plot(np.mean(test_mse_list, axis=0), label='test_mse')
        plt.xlabel('Epoch')
        plt.ylabel('MSE')
        plt.legend()
        plt.show()

class LogisticRegressionSGD:
    def __init__(self, learning_rate=0.001, max_iter=100):
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.weights = None
        self.bias = None
        self.dp = dataprocess('salary.csv')

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    def SGD(self, train_data, train_label, test_data, test_label, learning_rate=0.001, max_iter=1000, batch=1):

        n_samples, n_features = train_data.shape
        weights = np.zeros(n_features)
        bias = 0
        train_loss_list, test_loss_list = [], []
        for epoch in range(max_iter):

            for i in range(0, n_samples, batch):
                X_batch = train_data[i:i + batch]
                y_batch = train_label[i:i + batch]

                y_pred = np.dot(X_batch, weights) + bias
                y_pred = self.sigmoid(y_pred)

                gradient_w = -2 * np.dot(X_batch.T, (y_batch - y_pred)) / batch
                gradient_b = -2 * np.mean(y_batch - y_pred)

                weights -= learning_rate * gradient_w
                bias -= learning_rate * gradient_b

            train_loss = -np.mean(y_batch * np.log(y_pred + 1e-9) + (1 - y_batch) * np.log(1 - y_pred + 1e-9))
            train_loss_list.append(train_loss)

        test_pred = self.sigmoid(np.dot(test_data, weights) + bias)
        test_loss = -np.mean(test_label * np.log(test_pred + 1e-9) + (1 - test_label) * np.log(1 - test_pred + 1e-9))
        test_loss_list.append(test_loss)

            # if epoch % 10 == 0:
            #     print(f"Epoch {epoch + 1}/{max_iter}, Training Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}")

        return weights, bias, train_loss_list, test_loss_list

    def train(self, k=5):
        weight_list, bias_list,train_mse_list, test_mse_list = [], [], [], []
        for i in range(k):
            train_data, train_label, test_data, test_label = self.dp.task3_dataprocess(i)
            weight, bias, train_mse_error, test_mse_error = self.SGD(train_data, train_label, test_data, test_label, learning_rate=0.01, batch=1)

            weight_list.append(weight)
            bias_list.append(bias)
            train_mse_list.append(train_mse_error)
            test_mse_list.append(test_mse_error)

        print(test_mse_list)
        print("############################################")
        print('Logistic Regression function:')
        print("############################################")
        print("weight: ", np.mean(weight_list, axis=0))
        print("bias: ", np.mean(bias_list, axis=0))
        print("train_mse: ", np.mean(train_mse_list, axis=0))
        print("test_mse: ", np.mean(test_mse_list, axis=0))

        plt.plot(np.mean(train_mse_list, axis=0), label='train_mse')
        plt.plot(np.mean(test_mse_list, axis=0), label='test_mse')
        plt.xlabel('Epoch')
        plt.ylabel('MSE')
        plt.legend()
        plt.show()


# TODO: Implement the Decision Tree ID3 algorithm
class DecisionTreeID3:
    def __init__(self, epsilon=0.1):
        self.epsilon = epsilon
        self.tree = None

    def entropy(self, y):
        p = np.mean(y)
        if p == 0 or p == 1:
            return 0
        return -p * np.log2(p) - (1 - p) * np.log2(1 - p)

    def information_gain(self, X_col, y, threshold):
        left_idx = X_col <= threshold
        right_idx = X_col > threshold
        if len(y[left_idx]) == 0 or len(y[right_idx]) == 0:
            return 0

        parent_entropy = self.entropy(y)
        n = len(y)
        n_left, n_right = len(y[left_idx]), len(y[right_idx])
        child_entropy = (n_left / n) * self.entropy(y[left_idx]) + (n_right / n) * self.entropy(y[right_idx])
        return parent_entropy - child_entropy

    def best_split(self, X, y):
        best_gain = -1
        split_idx, split_threshold = None, None
        for i in range(X.shape[1]):
            thresholds = np.unique(X[:, i])
            for threshold in thresholds:
                gain = self.information_gain(X[:, i], y, threshold)
                if gain > best_gain:
                    best_gain = gain
                    split_idx, split_threshold = i, threshold
        return split_idx, split_threshold

    def grow_tree(self, X, y):
        if self.entropy(y) < self.epsilon or np.all(y == y[0]):
            return np.mean(y)

        split_idx, split_threshold = self.best_split(X, y)
        if split_idx is None:
            return np.mean(y)

        left_idx = X[:, split_idx] <= split_threshold
        right_idx = X[:, split_idx] > split_threshold

        left_tree = self.grow_tree(X[left_idx], y[left_idx])
        right_tree = self.grow_tree(X[right_idx], y[right_idx])

        return (split_idx, split_threshold, left_tree, right_tree)

    def train(self, X, y):
        self.tree = self.grow_tree(X, y)

    def predict_sample(self, x, tree):
        if isinstance(tree, float):
            return tree
        feature_idx, threshold, left_tree, right_tree = tree
        if x[feature_idx] <= threshold:
            return self.predict_sample(x, left_tree)
        else:
            return self.predict_sample(x, right_tree)

    def predict(self, X):
        return np.array([self.predict_sample(x, self.tree) for x in X])


if __name__ == '__main__':
    data_path = 'salary.csv'
    # dp = dataprocess(data_path)
    # dp.task1()
    # dp.task2()
    # p = Perceptron(data_path).train()
    LR = LogisticRegressionSGD(data_path).train()