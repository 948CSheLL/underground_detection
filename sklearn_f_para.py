import os
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import tree
from sklearn import svm
from sklearn import neighbors
from sklearn import ensemble
from sklearn.tree import ExtraTreeRegressor
from openpyxl import Workbook


class Inv_para_prediction():

    def __init__(self, X_filename="snr00withoutnoise.csv", y_filename="snr00withoutnoise_inv_para.csv", train_size=0.95):
        self.X_filename = "datas/" + X_filename
        self.y_filename = "datas/" + y_filename
        self.train_size = train_size
        # 通过read_csv来读取我们的目的数据集
        self.X_all = pd.read_csv(self.X_filename)
        self.y_all = pd.read_csv(self.y_filename)
        # 清洗不需要的数据
        # 下面的切片中前半部分选择想要的行数，后半部分选择想要的列数
        # 获得两个方向上的特征
        self.X_1 = self.X_all.iloc[:, 1:201]
        self.X_2 = self.X_all.iloc[:, 201:401]
        # 获得k1的标签
        self.para_dict = {}
        self.para_dict["k1"] = (self.X_1, self.y_all.iloc[:, 1:2])
        # 获得alpha1的标签
        self.para_dict["alpha1"] = (self.X_1.div(self.y_all.iloc[:, 1], axis="rows"), self.y_all.iloc[:, 2:3])
        # 获得beta1的标签
        self.para_dict["beta1"] = (self.X_1.div(self.y_all.iloc[:, 1], axis="rows"), self.y_all.iloc[:, 3:4])
        # 获得gamma1的标签
        self.para_dict["gamma1"] = (self.X_1.div(self.y_all.iloc[:, 1], axis="rows"), self.y_all.iloc[:, 4:5])
        # 获得k2的标签
        self.para_dict["k2"] = (self.X_2, self.y_all.iloc[:, 5:6])
        # 获得alpha2的标签
        self.para_dict["alpha2"] = (self.X_2.div(self.y_all.iloc[:, 5], axis="rows"), self.y_all.iloc[:, 6:7])
        # 获得beta2的标签
        self.para_dict["beta2"] = (self.X_2.div(self.y_all.iloc[:, 5], axis="rows"), self.y_all.iloc[:, 7:8])
        # 获得gamma2的标签
        self.para_dict["gamma2"] = (self.X_2.div(self.y_all.iloc[:, 5], axis="rows"), self.y_all.iloc[:, 8:9])
        self.model_dict = {}
        self.model_dict["decision_tree"] = tree.DecisionTreeRegressor()
        self.model_dict["linear_regression"] = LinearRegression()
        self.model_dict["svm_regression"] = svm.SVR()
        self.model_dict["k_neibour"] = neighbors.KNeighborsRegressor()
        # 使用20个决策树
        self.model_dict["random_forest"] = ensemble.RandomForestRegressor(n_estimators=20)
        # 这里使用50个决策树
        self.model_dict["adaboost_regression"] = ensemble.AdaBoostRegressor(n_estimators=50)
        # 这里使用100个决策树
        self.model_dict["gbrt_regression"] = ensemble.GradientBoostingRegressor(n_estimators=100)
        self.model_dict["bagging_regression"] = ensemble.BaggingRegressor()
        self.model_dict["extra_tree"] = ExtraTreeRegressor()
        self.dir = "images/" + datetime.now().strftime('%Y%m%d%H%M%S%f')
        for method_name in self.model_dict.keys():
            os.makedirs(self.dir + "/" + str(method_name))

    def write_into_excel(self, filename, head, datas_to_be_handled):
        work_book = Workbook()
        work_sheet = work_book.active
        work_sheet.append(head)
        for data in datas_to_be_handled:
            work_sheet.append(list(data))
        work_book.save(filename)

    def write_into_file(self, info):
        with open(self.dir + "/result.txt", "a") as file_obj:
            file_obj.write(info + "\n")

    def try_different_method(self, para_name, method_name):
        model = self.model_dict[method_name]
        X_train, X_test, y_train, y_test = train_test_split(
            self.para_dict[para_name][0], self.para_dict[para_name][1], train_size=self.train_size)
        model.fit(X_train, np.ravel(y_train))
        score = model.score(X_test, y_test)
        y_pred = model.predict(X_test)
        return y_pred, y_test.values, score

    def plot_parameters(self, para_name, method_name):
        y_pred, y_test, score = self.try_different_method(para_name, method_name)
        plt.figure()
        plt.plot(np.arange(len(y_test)), y_test, "go-", label=para_name + " True value")
        plt.plot(np.arange(len(y_pred)), y_pred, "ro-", label=para_name + " Predict value")
        plt.title(f"method:{method_name}---score:{score}")
        plt.legend(loc="best")
        plt.savefig(self.dir + "/" + method_name + "/" + para_name + "_predict.jpg")
        plt.show()
        return score

    def __call__(self):
        print("start!")
        model_names = self.model_dict.keys()
        head = ['para_name'] + list(model_names)
        datas_to_be_handle = []
        for para_name in self.para_dict.keys():
            data = []
            data.append(para_name)
            for method_name in model_names:
                score = self.plot_parameters(para_name, method_name)
                data.append(score)
            datas_to_be_handle.append(data)
        self.write_into_excel(self.dir + "/score_result.xls", head, datas_to_be_handle)
        print("done!")