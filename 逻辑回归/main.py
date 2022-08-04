#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
   This file contains the main program to read data, run the classifier,
   and print results to stdout.
   
   You can change the batch size and convergence threshold here. 

"""
import os
print('os.getcwd()',os.getcwd()) 
root_dir = os.path.dirname(__file__)  # 获取当前文件保存位置的绝对路径
os.chdir(root_dir) #设置根目录
print(os.getcwd())

import numpy as np
import random
from sklearn.model_selection import train_test_split
import pandas as pd
from models import LogisticRegression

ROOT_DIR_PREFIX = '.\\data\\' #我也不知道为啥子我的根目录需要再设置

DATA_FILE_NAME = 'normalized_data_binary.csv'


CENSUS_FILE_PATH = ROOT_DIR_PREFIX + DATA_FILE_NAME
print('CENSUS_FILE_PATH',CENSUS_FILE_PATH)

#一次次试，调整的参数
BATCH_SIZE = 100  #tune this parameter
CONV_THRESHOLD = 1e-7 #tune this parameter

#划分数据集
def import_census(file_path):
    '''
        Helper function to import the census dataset

        @param:
            train_path: path to census train data + labels
            test_path: path to census test data + labels
        @return:
            X_train: training data inputs
            Y_train: training data labels
            X_test: testing data inputs
            Y_test: testing data labels
    '''
    data = np.genfromtxt(file_path, delimiter=',', skip_header=False)
    X = data[:, :-1] #除了最后一列的所有为属性
    Y = data[:, -1].astype(int) #最后一列为Lable
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=0) #调包划分训练集和测试集，默认0.25，随机划分
    return X_train, Y_train, X_test, Y_test

#模型预测
def test_logreg():
    X_train, Y_train, X_test, Y_test = import_census(CENSUS_FILE_PATH)
    num_features = X_train.shape[1] #矩阵的列数，0为行数

    # Add a bias
    X_train_b = np.append(X_train, np.ones((len(X_train), 1)), axis=1)
    X_test_b = np.append(X_test, np.ones((len(X_test), 1)), axis=1)

    ### Logistic Regression ###调用models 文件
    model = LogisticRegression(num_features, BATCH_SIZE, CONV_THRESHOLD)
    num_epochs = model.train(X_train_b, Y_train) #模型中权重w的更新次数
    acc = model.accuracy(X_test_b, Y_test) * 100
    print("Test Accuracy: {:.1f}%".format(acc))
    print("Number of Epochs: " + str(num_epochs))

    return acc

#运行主程序   
def main():

    # Set random seeds. DO NOT CHANGE THIS IN YOUR FINAL SUBMISSION.
    random.seed(0) #设置随机数种子，数据划分也确定
    np.random.seed(0) #np生成X、Y确定

    acc_result = test_logreg()
    print(acc_result)

if __name__ == "__main__":
    main()

