import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, Activation,GRU
from nested_lstm import NestedLSTM
from sklearn.metrics import f1_score
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
level_num = 4
select_number = 10
min_max_scaler = preprocessing.MinMaxScaler()
import tensorflow as tf
# 加载真实故障数据
from keras.layers.convolutional import Conv1D
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

global num1
num1 =3
def load_data_old():
    filename_train = "E:\python\kong\CWGAN\data_set\Detect_Level4_train.csv"
    dataSet_train = pd.read_csv(filename_train, encoding='gbk')

    filename_test = "E:\python\kong\CWGAN\data_set\Detect_Level4_test.csv"
    dataSet_test = pd.read_csv(filename_test, encoding='gbk')


    dataSet_train = dataSet_train.values
    dataSet_test = dataSet_test.values

    # true_fault_data_train = dataSet_train[:, 1:]
    true_fault_data_train = dataSet_train[:, [6, 4, 48, 8, 18, 2, 28, 46, 30, 61, 5, 25, 32, 26, 9, 33]]
    true_fault_label_train = dataSet_train[:, 0]

    # true_fault_data_test = dataSet_test[:, 1:]
    true_fault_data_test = dataSet_test[:, [6, 4, 48, 8, 18, 2, 28, 46, 30, 61, 5, 25, 32, 26, 9, 33]]
    true_fault_label_test = dataSet_test[:, 0]
    #归一化
    true_fault_data_train = min_max_scaler.fit_transform(true_fault_data_train)
    true_fault_data_test = min_max_scaler.transform(true_fault_data_test)
    print("true_fault_data_train的大小:",true_fault_data_train.shape)
    print("true_fault_data_test的大小:",true_fault_data_test.shape)

    train_data1 = np.array(true_fault_data_train)
    train_label1 = np.array(true_fault_label_train)
    test_data1 = np.array(true_fault_data_test)
    test_label1 = np.array(true_fault_label_test)
    train_label1 = train_label1.reshape(3900,1)
    test_label1 = test_label1.reshape(1300,1)
    # print("train_data.shape", train_data1.shape)
    # print("train_label.shape", train_label1.shape)
    # print("test_data.shape", test_data1.shape)
    # print("test_label.shape", test_label1.shape)

#######################################################################################

    trX = train_data1[:, :]
    trY = train_label1[num1:, :]

    teX = test_data1[:, :]
    teY = test_label1[num1:, :]

    trainX = []
    for i in range(trX.shape[0] - num1):
        tempX = trX[i:i + num1, :]
        trainX.append(tempX)
    trainX = np.array(trainX)
    print("trainX.shape:", trainX.shape)

    trainY = trY[:, :]
    print(trainY.shape)

    testX = []
    for j in range(teX.shape[0] - num1):
        temX = teX[j:j + num1, :]
        testX.append(temX)
    testX = np.array(testX)
    print(testX.shape)

    testY = teY
    print("testY.shape:", testY.shape)

    print("trainY.shape", trainY.shape)
    print("trainX.shape", trainX.shape)
    print("testY.shape", testY.shape)
    print("testX.shape", testX.shape)

    return trainY, trainX, testY, testX

    # return train_label1,train_data1.reshape(1800,1,65),test_label1,test_data1.reshape(600,1,65)

def Lstm_models():
    train_label1,train_data1,test_label1,test_data1 = load_data_old()
    model = Sequential()
    model.add(LSTM(units = 64,input_shape=(num1,16),return_sequences=True,activation='relu'))
    model.add(LSTM(units = 64,return_sequences=False,activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(2,activation='softmax'))
    model.compile(loss='sparse_categorical_crossentropy', optimizer='Nadam', metrics=['accuracy'])
    model.fit(train_data1,train_label1,batch_size=20, epochs=20,verbose=2)
    a = np.argmax(model.predict(test_data1),axis=1)
    # print("Initial data classification results:\n",test_label1)
    print("LSTM classification results:\n",a)
    LSTM_AC = accuracy_score(test_label1, a)
    LSTM_f1 = f1_score(test_label1, a, average='macro')
    # print("LSTM_AC,LSTM_f1",LSTM_AC,LSTM_f1)
    return LSTM_AC,LSTM_f1
def GRU_models():
    train_label1,train_data1,test_label1,test_data1 = load_data_old()
    model = Sequential()
    model.add(GRU(units = 64,input_shape=(num1,16),return_sequences=True,activation='relu'))
    model.add(GRU(units = 64,return_sequences=False,activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(6,activation='softmax'))
    model.compile(loss='sparse_categorical_crossentropy', optimizer='Nadam', metrics=['accuracy'])
    model.fit(train_data1,train_label1,batch_size=20, epochs=20,verbose=2)
    a = np.argmax(model.predict(test_data1),axis=1)
    # print("Initial data classification results:\n",test_label1)
    print("GRU classification results:\n",a)
    GRU_AC = accuracy_score(test_label1, a)
    GRU_f1 = f1_score(test_label1, a, average='macro')
    # print("LSTM_AC,LSTM_f1",LSTM_AC,LSTM_f1)
    return GRU_AC,GRU_f1
def Nestlstm_models():
    train_label1,train_data1,test_label1,test_data1 = load_data_old()
    model1 = Sequential()
    # model1.add(NestedLSTM(64, depth=2, dropout=0, recurrent_dropout=0.2,activation='relu',return_sequences=True))
    # model1.add(NestedLSTM(64, depth=2, dropout=0, recurrent_dropout=0,activation='relu',return_sequences=True))
    # model1.add(NestedLSTM(64, depth=2, dropout=0, recurrent_dropout=0,activation='relu',return_sequences=True))
    model1.add(NestedLSTM(64, depth=2, dropout=0, recurrent_dropout=0.2,activation='relu'))
    model1.add(Dense(2,activation='softmax'))
    model1.compile(loss='sparse_categorical_crossentropy', optimizer='Nadam', metrics=['accuracy'])
    model1.fit(train_data1,train_label1,batch_size=20, epochs=20,verbose=2)
    a = np.argmax(model1.predict(test_data1),axis=1)
    # print("Initial data classification results:\n",test_label1)
    print("NLSTM classification results:\n",a)
    NLSTM_AC = accuracy_score(test_label1, a)
    NLSTM_f1 = f1_score(test_label1, a, average='macro')
    # print("LSTM_AC,LSTM_f1",LSTM_AC,LSTM_f1)
    return NLSTM_AC,NLSTM_f1
Lstm_models()