#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
   This file contains the Logistic Regression classifier

'''
import random
import numpy as np

#LogisticRegression 模型实现
class LogisticRegression:
    '''
    Two-class Logistic Regression that learns weights using 二分类逻辑回归
    stochastic gradient descent. 梯度下降算法SGD
    '''
#定义类
    def __init__(self, n_features, batch_size, conv_threshold):
        '''
        Initializes a LogisticRegression classifer.

        @attrs:
            n_features: the number of features in the classification problem 特征个数 len(X_train) X_train.shape[1]
            weights: The weights of the Logistic Regression model 特征的权重
            alpha: The learning rate used in stochastic gradient descent 学习率
        '''
        self.n_features = n_features
        self.weights = np.zeros(n_features + 1)  # An extra weight added for the bias   X_train_b
        self.alpha = 0.03  # DO NOT TUNE THIS PARAMETER
        self.batch_size = batch_size
        self.conv_threshold = conv_threshold

#梯度下降算法训练模型参数 num_epochs = model.train(X_train_b, Y_train)
    def train(self, X, Y):
        '''
        Trains the model using stochastic gradient descent

        @params:
            X: a 2D Numpy array where each row contains an example, padded by 1 column for the bias
            Y: a 1D Numpy array containing the corresponding labels for each example
        @return:
            num_epochs: integer representing the number of epochs taken to reach convergence
        '''
        times = 0 #更新次数
        while True:
            times += 1
            Rand = np.random.randint(0, len(X), size=self.batch_size) #随机选择每个批量
            X_new = X[Rand]
            Y_new = Y[Rand]
            loss_old = self.loss(X, Y) #计算损失
            weights_old = self.weights
            #计算梯度
            gradients = (1 / (1 + np.exp(-self.weights @ X_new.T)) - Y_new) @ X_new # @矩阵乘法，np.dot
            #更新w
            self.weights = weights_old - self.alpha * gradients
            if abs(self.loss(X, Y) - loss_old) < self.conv_threshold or times > 1e4: #迭代太多次了
                print('Rand',Rand)
                break
        return times

#损失函数
    def loss(self, X, Y):
        '''
        Returns the total log loss on some dataset (X, Y), divided by the number of examples.
        @params:
            X: 2D Numpy array where each row contains an example, padded by 1 column for the bias
            Y: 1D Numpy array containing the corresponding labels for each example
        @return:
            A float number which is the average loss of the model on the dataset
        '''
        H = 1 / (1 + np.exp((-1) * np.dot(X, self.weights)))
        loss = - np.dot(Y, np.log(H)) - np.dot((1 - Y), np.log(1 - H))
        return loss / len(X)

    def predict(self, X):
        '''
        Compute predictions based on the learned weigths and examples X

        @params:
            X: a 2D Numpy array where each row contains an example, padded by 1 column for the bias
        @return:
            A 1D Numpy array with one element for each row in X containing the predicted class.
        '''
        H = 1 / (1 + np.exp((-1) * np.dot(X, self.weights)))
        H[H > 0.5] = 1
        H[H < 0.5] = 0

        return H

    def accuracy(self, X, Y):
        '''
        Outputs the accuracy of the trained model on a given testing dataset X and labels Y.

        @params:
            X: a 2D Numpy array where each row contains an example, padded by 1 column for the bias
            Y: a 1D Numpy array containing the corresponding labels for each example
        @return:
            a float number indicating accuracy (between 0 and 1)
        '''
        Y_new = self.predict(X)
        acc = (Y_new == Y).mean()
        return acc