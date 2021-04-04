# 不使用GAN扩充故障训练样本,直接使用原始故障样本做诊断

# -*- coding: utf-8 -*-
"""
Created on  June  5 10:54:44 2019

@author: Jianye Su
"""

from lstm_softmax_super import Lstm_models,Nestlstm_models,GRU_models
import numpy as np
import scipy.io as sio
import random
from sklearn import svm
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from sklearn import preprocessing
from lightgbm import LGBMClassifier
from sklearn.ensemble import AdaBoostClassifier
level_num = 4
select_number = 10
min_max_scaler = preprocessing.MinMaxScaler()

# 加载真实故障数据

def load_data(num):
    filename_train = "data_set//Level1_train.csv"
    dataSet_train = pd.read_csv(filename_train, encoding='gbk')

    filename_test = "data_set//Level1_test.csv"
    dataSet_test = pd.read_csv(filename_test, encoding='gbk')

    # dataSet_train = dataSet_train.values
    dataSet_test = dataSet_test.values
    print("train_dataAndlabel的类型：", type(dataSet_train))

    # np.random.shuffle(dataSet_train)  # 打乱行的顺序
    # np.random.shuffle(dataSet_test)  # 打乱行的顺序

    print(dataSet_test)
    true_fault_data_train = dataSet_train.iloc[:, 1:]
    true_fault_label_train = dataSet_train.iloc[num:, 0]

    true_fault_data_test = dataSet_test[:200, 1:]
    true_fault_label_test = dataSet_test[num:200, 0]

    train_data = true_fault_data_train
    train_label = true_fault_label_train
    test_data = true_fault_data_test
    test_label = true_fault_label_test

    train_data = np.array(train_data)
    train_label = np.array(train_label)
    test_data = np.array(test_data)
    test_label = np.array(test_label)

    # print("train_data.shape", train_data.shape)
    # print("train_label.shape", train_label.shape)
    # print("test_data.shape", test_data.shape)
    # print("test_label.shape", test_label.shape)
    trainX_data = []
    testX_data = []
    for j in range(train_data.shape[0]-num):
        trainX = train_data[j:j + num, :]
        trainX_data.append(trainX)
    trainX_data = np.array(trainX_data)
    print(trainX_data.shape)
    for j in range(test_data.shape[0]-num):
        testX = test_data[j:j + num, :]
        testX_data.append(testX)
    testX_data = np.array(testX_data)
    print(testX_data.shape)
    trainX_data = np.reshape(trainX_data, (trainX_data.shape[0], trainX_data.shape[1], testX_data.shape[2]))
    testX_data = np.reshape(testX_data, (testX_data.shape[0], testX_data.shape[1], testX_data.shape[2]))
    print("trainX_data",trainX_data.shape)
    print("testX_data", testX_data.shape)
    print("train_label", train_label.shape)
    print("test_label", test_label.shape)
    return train_label,trainX_data,test_label,testX_data
    # return train_data1,train_label1,test_data1,test_label1


def load_data_old():
    filename_train = "data_set//Level1_train.csv"
    # filename_train = "data_set_1000train\Level1_train.csv"
    dataSet_train = pd.read_csv(filename_train, encoding='gbk')

    filename_test = "data_set//Level1_test.csv"
    # filename_test = "data_set_1000train\Level1_test.csv"
    dataSet_test = pd.read_csv(filename_test, encoding='gbk')


    dataSet_train = dataSet_train.values
    dataSet_test = dataSet_test.values
    # print("train_dataAndlabel的类型：", type(dataSet_train))

    # np.random.shuffle(dataSet_train)  # 打乱行的顺序
    # np.random.shuffle(dataSet_test)  # 打乱行的顺序

    # print(dataSet_test)
    true_fault_data_train = dataSet_train[:, [6,4,48,8,18,2,28,46,30,61,5,25,32,26,9,33]]
    # true_fault_data_train = dataSet_train[:, [6, 4, 48, 8, 18, 2, 28, 46]]
    true_fault_label_train = dataSet_train[:, 0]

    true_fault_data_test = dataSet_test[400:500, [6,4,48,8,18,2,28,46,30,61,5,25,32,26,9,33]]
    # true_fault_data_test = dataSet_test[:, [6, 4, 48, 8, 18, 2, 28, 46]]
    true_fault_label_test = dataSet_test[400:500, 0]
    #归一化
    true_fault_data_train = min_max_scaler.fit_transform(true_fault_data_train)
    true_fault_data_test = min_max_scaler.fit_transform(true_fault_data_test)


    train_data1 = np.array(true_fault_data_train)
    train_label1 = np.array(true_fault_label_train)
    test_data1 = np.array(true_fault_data_test)
    test_label1 = np.array(true_fault_label_test)



    return train_label1,train_data1,test_label1,test_data1

def classify(train_data, train_label):
    train_label = train_label.ravel()#将多维数据降成一维
    # LSTM_AC, LSTM_f1 = Lstm_models()
    # GRU_AC, GRU_f1 = GRU_models()
    NLSTM_AC, NLSTM_f1 = Nestlstm_models()

    lgbmModel = LGBMClassifier(max_depth=5,num_leaves=25,learning_rate=0.007,n_estimators=1000,min_child_samples=80,
    subsample=0.8,colsample_bytree=1,reg_alpha=0,reg_lambda=0,random_state=np.random.randint(10e6))
    lgbmModel.fit(train_data, train_label)
    lgbm_pre = lgbmModel.predict(test_data)

    lgbm_AC = accuracy_score(test_label, lgbm_pre)
    lgbm_f1 = f1_score(test_label, lgbm_pre, average='macro')

    AdaBoostModel = AdaBoostClassifier(base_estimator=None,n_estimators=50,learning_rate=1,algorithm='SAMME.R',random_state=None)
    AdaBoostModel.fit(train_data, train_label)
    AdaBoost_pre = AdaBoostModel.predict(test_data)
    AdaBoost_AC = accuracy_score(test_label, AdaBoost_pre)
    AdaBoost_f1 = f1_score(test_label, AdaBoost_pre, average='macro')

    rfc1 = RandomForestClassifier(n_estimators=40, max_depth=None, min_samples_split=2, random_state=2)#随机森林分类器
    rfc1.fit(train_data, train_label)
    RF_pre = rfc1.predict(test_data)
    RF_AC = accuracy_score(test_label, RF_pre)
    RF_f1 = f1_score(test_label, RF_pre, average='macro')


    clf = SVC(kernel='rbf', C=9, gamma=0.1)
    clf.set_params(kernel='rbf', probability=True).fit(train_data, train_label)#set_params：设置SVC函数的参数
    clf.predict(train_data)
    test_pre = clf.predict(test_data)
    SVM_AC = accuracy_score(test_label, test_pre)
    SVM_f1 = f1_score(test_label, test_pre, average='macro')

    # decision tree
    dtc = DecisionTreeClassifier()
    dtc.fit(train_data, train_label)
    dt_pre = dtc.predict(test_data)
    DT_AC = accuracy_score(test_label, dt_pre)
    DT_f1 = f1_score(test_label, dt_pre, average='macro')


    MLP = MLPClassifier(solver='lbfgs', alpha=1e-4,
                        hidden_layer_sizes=(100, 3), random_state=1)
    MLP.fit(train_data, train_label)
    MLP_predict = MLP.predict(test_data)
    MLP_AC = accuracy_score(test_label, MLP_predict)
    MLP_f1 = f1_score(test_label, MLP_predict, average='macro')

    # KNN
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(train_data, train_label)
    knn_predict = knn.predict(test_data)
    KNN_AC = accuracy_score(test_label, knn_predict)
    KNN_f1 = f1_score(test_label, knn_predict, average='macro')

    # LogisticRegression
    classifier = LogisticRegression()
    classifier.fit(train_data, train_label)
    lg_predict = classifier.predict(test_data)
    LG_AC = accuracy_score(test_label, lg_predict)
    LG_f1 = f1_score(test_label, lg_predict, average='macro')
    #
    # print("===== Diagnosis original=======")
    # print('Original Accuracy:')
    # print(RF_AC, SVM_AC, DT_AC, NB_AC, MLP_AC, KNN_AC, LG_AC)
    # print('F1-score')
    # print(RF_f1, SVM_f1, DT_f1, NB_f1, MLP_f1, KNN_f1, LG_f1)
    # Main.py按照original.py, Ensemble.py, vae_od.py顺序执行，结果依次存入下面文件
#     file_name1 = "./temp_result/Diagnosis_"+str(select_number)+"Level"+str(level_num)+"_Accuracy_result.txt"
#     file_name2 = "./temp_result/Diagnosis_"+str(select_number)+"Level"+str(level_num)+"_f1_score_result.txt"
#     with open(file_name1, "a") as f:
#         f.writelines([str(RF_AC), ' ', str(SVM_AC), ' ', str(DT_AC), ' ', str(NB_AC), ' ', str(MLP_AC), ' ', str(KNN_AC), ' ', str(LG_AC), '\n'])
#     with open(file_name2, "a") as f:
#         f.writelines([str(RF_f1), ' ', str(SVM_f1), ' ', str(DT_f1), ' ', str(NB_f1), ' ', str(MLP_f1), ' ', str(KNN_f1), ' ', str(LG_f1), '\n'])
#     return NLSTM_AC,LSTM_AC ,GRU_AC,lgbm_AC,AdaBoost_AC,RF_AC, SVM_AC, DT_AC, MLP_AC, KNN_AC, LG_AC,NLSTM_f1,LSTM_f1,GRU_f1,lgbm_f1,AdaBoost_f1,RF_f1,SVM_f1,DT_f1,MLP_f1,KNN_f1,LG_f1
    return NLSTM_AC,lgbm_AC,AdaBoost_AC,RF_AC, SVM_AC, DT_AC, MLP_AC, KNN_AC, LG_AC,NLSTM_f1,lgbm_f1,AdaBoost_f1,RF_f1,SVM_f1,DT_f1,MLP_f1,KNN_f1,LG_f1
#
# if __name__ == "__main__":
# train_label,train_data,test_label,test_data = load_data(3)
train_label,train_data,test_label,test_data = load_data_old()
# NLSTM_AC,LSTM_AC ,GRU_AC,lgbm_AC,AdaBoost_AC,RF_AC, SVM_AC, DT_AC, MLP_AC, KNN_AC, LG_AC,NLSTM_f1,LSTM_f1,GRU_f1,lgbm_f1,AdaBoost_f1,RF_f1,SVM_f1,DT_f1,MLP_f1,KNN_f1,LG_f1 = classify(train_data, train_label)
NLSTM_AC,lgbm_AC,AdaBoost_AC,RF_AC, SVM_AC, DT_AC, MLP_AC, KNN_AC, LG_AC,NLSTM_f1,lgbm_f1,AdaBoost_f1,RF_f1,SVM_f1,DT_f1,MLP_f1,KNN_f1,LG_f1 = classify(train_data, train_label)
# for i in range(3):
#
#     LSTM_AC ,RF_AC, SVM_AC, DT_AC, MLP_AC, KNN_AC, LG_AC = classify(train_data, train_label)
#     if i == 0:
#         ave_LSTM = LSTM_AC[1]
#     else:
#         ave_LSTM = np.append(ave_LSTM, LSTM_AC[1])
# print(LSTM_AC)
print("NLSTM accuracy:" + str(NLSTM_AC))
print("NLSTM F1-score:" + str(NLSTM_f1))
print("=============")
print("=============")
# print("LSTM accuracy:" + str(LSTM_AC))
# print("LSTM F1-score:" + str(LSTM_f1))
# print("=============")
# print("GRU accuracy:" + str(GRU_AC))
# print("GRU F1-score:" + str(GRU_f1))
print("=============")
print("LGBM accuracy:" + str(lgbm_AC))
print("LGBM F1-score:" + str(lgbm_f1))
print("=============")
print("AdaBoost accuracy:" + str(AdaBoost_AC))
print("AdaBoost F1-score:" + str(AdaBoost_f1))
print("=============")
print("Random forest accuracy:" + str(RF_AC))
print("Random F1-score:" + str(RF_f1))
print("=============")
print("SVM accuracy:" + str(SVM_AC))
print("SVM F1-score:" + str(SVM_f1))
print("=============")
print("Decision tree accuracy:" + str(DT_AC))
print("Decision tree F1-score:" + str(DT_f1))
print("=============")
print("=============")
print("Multilayer perceptron accuracy:" + str(MLP_AC))
print("Multilayer F1-score:" + str(MLP_f1))
print("=============")
print("KNN accuracy:" + str(KNN_AC))
print("KNN F1-score:" + str(KNN_f1))
print("=============")
print("LogisticRegression accuracy:" + str(LG_AC))
print("LogisticRegression F1-score:" + str(LG_f1))
