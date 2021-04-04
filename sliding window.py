import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, Activation,GRU,Bidirectional,TimeDistributed,BatchNormalization,Multiply
from nested_lstm import NestedLSTM
from nested_lstm_True import NestedLSTM_True
import pandas as pd
from sklearn.metrics import accuracy_score,f1_score
from sklearn import preprocessing
from keras.models import *
from keras.layers import merge
from keras.layers.core import *
from keras.models import Sequential
from keras.layers import Dense,LSTM,MaxPooling1D,Dropout,AveragePooling1D,concatenate
from keras.layers.convolutional import Conv1D
from keras import optimizers
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.metrics import mean_squared_error,mean_absolute_error
import matplotlib.pyplot as plt
from keras.models import load_model
# from pgmpy.models import BayesianModel
level_num = 4
select_number = 10
min_max_scaler = preprocessing.MinMaxScaler()
import tensorflow as tf
# 加载真实故障数据
from keras.layers import Conv1D,Conv2D,MaxPool2D
global num1

global lenth
lenth = 65
global size_batch
size_batch = 64


def load_data_old():
    filename_train = "data_set_5000train\Level4_train.csv"
    # filename_train = "data_set\Level2_train.csv"
    dataSet_train = pd.read_csv(filename_train, encoding='gbk')

    filename_test = "data_set_5000train\Level4_train.csv"
    # filename_test = "data_set\Level2_test.csv"
    dataSet_test = pd.read_csv(filename_test, encoding='gbk')

    dataSet_train = dataSet_train.values
    dataSet_test = dataSet_test.values
    #
    train1 = dataSet_train[0:4000,:]
    test1 = dataSet_test[4000:4500, :]
    train2 = dataSet_train[5000:9000, :]
    test2= dataSet_test[9000:9500, :]
    train3 = dataSet_train[10000:14000, :]
    test3= dataSet_test[14000:14500, :]
    train4 = dataSet_train[15000:19000, :]
    test4 = dataSet_test[19000:19500, :]
    train5 = dataSet_train[20000:24000, :]
    test5 = dataSet_test[24000:24500, :]
    train6 = dataSet_train[25000:29000, :]
    test6 = dataSet_test[29000:29500, :]
    train7 = dataSet_train[30000:34000, :]
    test7 = dataSet_test[34000:34500, :]
    train8 = dataSet_train[35000:39000, :]
    test8 = dataSet_test[39000:39500, :]

    true_fault_data_train1 = np.concatenate((train1,train2,train3,train4,train5,train6,train7,train8))
    true_fault_data_test1 = np.concatenate((test1,test2,test3,test4,test5,test6,test7,test8))
    print(true_fault_data_train1.shape)
    print(true_fault_data_test1.shape)


    # true_fault_data_train = true_fault_data_train1[:, [6,4,48,8,18,2,28,46,30,61,5,25,32,26,9,33]]
    true_fault_data_train = true_fault_data_train1[:, 1:]
    # true_fault_data_train = dataSet_train[:, 1:]
    true_fault_label_train = true_fault_data_train1[:, 0]

    # true_fault_data_test = true_fault_data_test1[:, [6,4,48,8,18,2,28,46,30,61,5,25,32,26,9,33]]
    true_fault_data_test = true_fault_data_test1[:,1:]
    # true_fault_data_test = dataSet_test[:, 1:]
    true_fault_label_test = true_fault_data_test1[:, 0]
    #归一化
    true_fault_data_train = min_max_scaler.fit_transform(true_fault_data_train)
    true_fault_data_test = min_max_scaler.transform(true_fault_data_test)
    # print("true_fault_data_train的大小:",true_fault_data_train.shape)
    # print("true_fault_data_test的大小:",true_fault_data_test.shape)

    train_data1 = np.array(true_fault_data_train)
    train_label1 = np.array(true_fault_label_train)
    test_data1 = np.array(true_fault_data_test)
    test_label1 = np.array(true_fault_label_test)
    train_label1 = train_label1.reshape(32000,1)
    test_label1 = test_label1.reshape(4000,1)
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



# 注意力机制的另一种写法 适合上述报错使用 来源:https://blog.csdn.net/uhauha2929/article/details/80733255
def attention_3d_block2(inputs, single_attention_vector=True):
    # 如果上一层是LSTM，需要return_sequences=True
    # inputs.shape = (batch_size, time_steps, input_dim)
    time_steps = K.int_shape(inputs)[1]
    input_dim = K.int_shape(inputs)[2]
    a = Permute((2, 1))(inputs)
    a = Dense(time_steps, activation='softmax')(a)
    if single_attention_vector:
        a = Lambda(lambda x: K.mean(x, axis=1))(a)
        a = RepeatVector(input_dim)(a)#RepeatVector不改变我们的步长，改变我们的每一步的维数（即：属性长度）
    a_probs = Permute((2, 1))(a)
    # 乘上了attention权重，但是并没有求和，好像影响不大
    # 如果分类任务，进行Flatten展开就可以了
    # element-wise
    output_attention_mul = Multiply()([inputs, a_probs])
    return output_attention_mul

def Lstm_models():
    train_label1,train_data1,test_label1,test_data1 = load_data_old()
    print("LSTM Processing:")
    model = Sequential()
    model.add(LSTM(units = 64,input_shape=(num1,lenth),return_sequences=False,activation='relu'))
    model.add(Dropout(0.2))
    # model.add(LSTM(units = 64,return_sequences=False,activation='relu'))
    model.add(Dense(8,activation='softmax'))
    model.compile(loss='sparse_categorical_crossentropy', optimizer='Nadam', metrics=['accuracy'])
    model.fit(train_data1,train_label1,batch_size=size_batch, epochs=30,verbose=2)
    a = np.argmax(model.predict(test_data1),axis=1)
    # print("Initial data classification results:\n",test_label1)
    # print("LSTM classification results:\n",a)
    LSTM_AC = accuracy_score(test_label1, a)
    LSTM_f1 = f1_score(test_label1, a, average='macro')
    print("LSTM_AC,LSTM_f1",LSTM_AC,LSTM_f1)
    return LSTM_AC,LSTM_f1

def GRU_models():
    train_label1,train_data1,test_label1,test_data1 = load_data_old()
    print("GRU Processing:")
    model = Sequential()
    model.add(GRU(units = 64,input_shape=(num1,lenth),return_sequences=False,activation='relu'))
    model.add(Dropout(0.2))
    # model.add(GRU(units = 64,return_sequences=False,activation='relu'))
    model.add(Dense(8,activation='softmax'))
    model.compile(loss='sparse_categorical_crossentropy', optimizer='Nadam', metrics=['accuracy'])
    model.fit(train_data1,train_label1,batch_size=size_batch, epochs=30,verbose=2)
    a = np.argmax(model.predict(test_data1),axis=1)
    # print("Initial data classification results:\n",test_label1)
    # print("GRU classification results:\n",a)
    GRU_AC = accuracy_score(test_label1, a)
    GRU_f1 = f1_score(test_label1, a, average='macro')
    print("GRU_AC,GRU_f1",GRU_AC,GRU_f1)
    return GRU_AC,GRU_f1

def Bilstm_models():
    train_label1,train_data1,test_label1,test_data1 = load_data_old()
    print("Bilstm Processing:")
    # train_label1, train_data1, test_label1, test_data1 = load_data()
    model1 = Sequential()
    # model1.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu', input_shape=(num1,lenth)))
    # model1.add(Bidirectional(LSTM(units=64, return_sequences=False, activation='relu')))
    model1.add(Bidirectional(LSTM(units=64, input_shape=(num1, lenth), return_sequences=False, activation='relu')))
    model1.add(Dense(8,activation='softmax'))
    model1.compile(loss='sparse_categorical_crossentropy', optimizer='Nadam', metrics=['accuracy'])
    model1.fit(train_data1,train_label1,batch_size=size_batch , epochs=30,verbose=2)#
    a = np.argmax(model1.predict(test_data1),axis=1)
    np.set_printoptions(threshold=100000000)
    # print("BiLSTM classification results:\n",a)
    BiLSTM_AC = accuracy_score(test_label1, a)
    BiLSTM_f1 = f1_score(test_label1, a, average='macro')
    print("BiLSTM_AC,BiLSTM_f1",BiLSTM_AC,BiLSTM_f1)
    return BiLSTM_AC,BiLSTM_f1

def BiLSTM_duo():
    train_label1, train_data1, test_label1, test_data1 = load_data_old()
    print("BiLSTM_duo Processing:")
    inputs_left = Input(shape=(num1,lenth),name='inp_left')
    conv_left = Conv1D(filters=32, kernel_size=5, padding='same', activation='relu')(inputs_left)

    inputs_right = Input(shape=(num1,lenth),name='inp_right')
    conv_inputs_right0 = Conv1D(filters=32, kernel_size=5, padding='same', activation='relu')(inputs_right)
    conv_inputs_right = Conv1D(filters=32, kernel_size=5, padding='same', activation='relu')(conv_inputs_right0)

    # inputs_right1 = Input(shape=(num1,lenth),name='inp_right')
    # conv_inputs_right0 = Conv1D(filters=32, kernel_size=3, padding='same', activation='relu')(inputs_right1)
    # conv_inputs_right = Conv1D(filters=32, kernel_size=3, padding='same', activation='relu')(conv_inputs_right0)
    # conv_inputs_right1 = Conv1D(filters=32, kernel_size=3, padding='same', activation='relu')(conv_inputs_right)
    output_attention_mul_right = merge.average([conv_left,conv_inputs_right])
    out0 = Bidirectional(LSTM(units=64, return_sequences=False, activation='relu'))(output_attention_mul_right)
    dp = Dropout(0.2, name='dp0')(out0)
    output = Dense(8, activation='softmax')(dp)
    model = Model(input=[inputs_left,inputs_right], output=output)
    model.compile(loss='sparse_categorical_crossentropy', optimizer='Nadam', metrics=['accuracy'])
    model.fit([train_data1,train_data1], train_label1, batch_size=64 , epochs=30,verbose=2)
    a = np.argmax(model.predict([test_data1,test_data1]), axis=1)
    np.set_printoptions(threshold=100000000)
    # print("BiLSTM_duo classification results:\n", a)
    BiLSTM_AC = accuracy_score(test_label1, a)
    BiLSTM_f1 = f1_score(test_label1, a, average='macro')
    print("BiLSTM_duo_AC,BiLSTM_duo_f1",BiLSTM_AC,BiLSTM_f1)
    return BiLSTM_AC, BiLSTM_f1

def BiLSTM_attention():
    train_label1, train_data1, test_label1, test_data1 = load_data_old()
    print("BiLSTM_attention Processing:")
    ############    model_left  ########
    inputs_left = Input(shape=(num1,lenth),name='inp_left')
    # Conv_left = Conv1D(filters=32, kernel_size=5, padding='same', activation='relu')(inputs_left)
    lstm_out_left = Bidirectional(LSTM(units = 64,return_sequences=True,activation='relu'))(inputs_left)
    a_left = Permute((2, 1), name='permute0_left')(lstm_out_left)
    a_left = Dense(num1, activation='softmax', name='dense_left')(a_left)
    a_left_probs = Permute((2, 1), name='permute1_left')(a_left)
    output_attention_mul_left = merge.concatenate([lstm_out_left, a_left_probs], name='attention_mul_left')
    attention_mul_left = Flatten(name='flatten_left')(output_attention_mul_left)
    dp = Dropout(0, name='dp0')(attention_mul_left)
    output = Dense(8, activation='softmax')(dp)
    model = Model(input=inputs_left, output=output)
    model.compile(loss='sparse_categorical_crossentropy', optimizer='Nadam', metrics=['accuracy'])
    model.fit(train_data1, train_label1, batch_size=size_batch , epochs=30,verbose=2)
    a = np.argmax(model.predict(test_data1), axis=1)
    np.set_printoptions(threshold=100000000)
    # print("BiLSTM_attention classification results:\n", a)
    BiLSTM_attention_AC = accuracy_score(test_label1, a)
    BiLSTM_attention_f1 = f1_score(test_label1, a, average='macro')
    print("BiLSTM_attention_AC,BiLSTM_attention_f1",BiLSTM_attention_AC,BiLSTM_attention_f1)
    return BiLSTM_attention_AC, BiLSTM_attention_f1

def BiLSTM_duo_attention():
    train_label1, train_data1, test_label1, test_data1 = load_data_old()
    print("BiLSTM_duo_attention Processing:")
    inputs_left = Input(shape=(num1,lenth),name='inp_left')
    conv_left = Conv1D(filters=32, kernel_size=5, padding='same', activation='relu')(inputs_left)

    inputs_right = Input(shape=(num1,lenth),name='inp_right')
    conv_inputs_right0 = Conv1D(filters=32, kernel_size=5, padding='same', activation='relu')(inputs_right)
    conv_inputs_right = Conv1D(filters=32, kernel_size=5, padding='same', activation='relu')(conv_inputs_right0)
    # a_right = Permute((2, 1), name='permute0_right')(conv_inputs_right)
    # a_right = Dense(4, activation='softmax', name='dense_right')(a_right)
    # a_right_probs = Permute((2, 1), name='permute1_right')(a_right)
    # inputs_right1 = Input(shape=(num1,lenth),name='inp_right')
    # conv_inputs_right0 = Conv1D(filters=32, kernel_size=3, padding='same', activation='relu')(inputs_right1)
    # conv_inputs_right = Conv1D(filters=32, kernel_size=3, padding='same', activation='relu')(conv_inputs_right0)
    # conv_inputs_right1 = Conv1D(filters=32, kernel_size=3, padding='same', activation='relu')(conv_inputs_right)
    output_attention_mul_right = merge.average([conv_left,conv_inputs_right])
    out0 = Bidirectional(LSTM(units=64, return_sequences=True, activation='relu'))(output_attention_mul_right)
    a_left0 = Permute((2, 1), name='permute0_left')(out0)
    a_left = Dense(num1, activation='relu', name='dense_left')(a_left0)
    a_left_probs = Permute((2, 1), name='permute1_left')(a_left)
    output_attention_mul_left = merge.average([out0, a_left_probs], name='attention_mul_left')
    attention_mul_left = Flatten(name='flatten_left')(output_attention_mul_left)
    # dp = Dropout(0.2, name='dp0')(attention_mul_left)
    output1 = Dense(8, activation='softmax')(attention_mul_left)
    model = Model(input=[inputs_left,inputs_right], output=output1 )
    model.compile(loss='sparse_categorical_crossentropy', optimizer='Nadam', metrics=['accuracy'])
    model.fit([train_data1,train_data1], train_label1, batch_size=64 , epochs=30,verbose=2)
    a = np.argmax(model.predict([test_data1,test_data1]), axis=1)
    np.set_printoptions(threshold=100000000)
    # print("BiLSTM_duo_attention classification results:\n", a)
    BiLSTM_AC = accuracy_score(test_label1, a)
    BiLSTM_f1 = f1_score(test_label1, a, average='macro')
    print("BiLSTM_duo_attention_AC,BiLSTM_duo_attention_f1",BiLSTM_AC,BiLSTM_f1)
    return BiLSTM_AC, BiLSTM_f1


def Nestlstm_models():
    train_label1,train_data1,test_label1,test_data1 = load_data_old()
    print("Nestlstm_models Processing:")
    # train_label1, train_data1, test_label1, test_data1 = load_data()
    model1 = Sequential()
    # model1.add(Conv1D(filters=32, kernel_size=5, padding='same', activation='relu', input_shape=(num1, lenth)))
    # model1.add(NestedLSTM(32, depth=2, dropout=0, recurrent_dropout=0,activation='relu',return_sequences=True))
    # model1.add(Dropout(0.2))
    model1.add(NestedLSTM(64, depth=2, dropout=0, recurrent_dropout=0.2,activation='relu'))
    model1.add(Dense(8,activation='softmax'))
    model1.compile(loss='sparse_categorical_crossentropy', optimizer='Nadam', metrics=['accuracy'])
    model1.fit(train_data1,train_label1,batch_size=size_batch , epochs=30,verbose=2)#
    a = np.argmax(model1.predict(test_data1),axis=1)
    # print("Initial data classification results:\n",test_label1)
    np.set_printoptions(threshold=100000000)
    # print("NLSTM classification results:\n",a)
    NLSTM_AC = accuracy_score(test_label1, a)
    NLSTM_f1 = f1_score(test_label1, a, average='macro')
    print("NLSTM_AC,NLSTM_f1",NLSTM_AC,NLSTM_f1)
    return NLSTM_AC,NLSTM_f1

def NLSTM_duo():
    train_label1, train_data1, test_label1, test_data1 = load_data_old()
    print("NLSTM_duo Processing:")
    inputs_left = Input(shape=(num1,lenth),name='inp_left')
    conv_left = Conv1D(filters=32, kernel_size=5, padding='same', activation='relu')(inputs_left)

    inputs_right = Input(shape=(num1,lenth),name='inp_right')
    conv_inputs_right0 = Conv1D(filters=32, kernel_size=5, padding='same', activation='relu')(inputs_right)
    conv_inputs_right = Conv1D(filters=32, kernel_size=5, padding='same', activation='relu')(conv_inputs_right0)
    # a_right = Permute((2, 1), name='permute0_right')(conv_inputs_right)
    # a_right = Dense(4, activation='softmax', name='dense_right')(a_right)
    # a_right_probs = Permute((2, 1), name='permute1_right')(a_right)
    # inputs_right1 = Input(shape=(num1,lenth),name='inp_right')
    # conv_inputs_right0 = Conv1D(filters=32, kernel_size=3, padding='same', activation='relu')(inputs_right1)
    # conv_inputs_right = Conv1D(filters=32, kernel_size=3, padding='same', activation='relu')(conv_inputs_right0)
    # conv_inputs_right1 = Conv1D(filters=32, kernel_size=3, padding='same', activation='relu')(conv_inputs_right)
    output_attention_mul_right = merge.concatenate([conv_left,conv_inputs_right])
    out0 = NestedLSTM(64, depth=2, dropout=0, recurrent_dropout=0.2,activation='relu')(output_attention_mul_right)
    dp = Dropout(0.2, name='dp0')(out0)
    output = Dense(8, activation='softmax')(dp)
    model = Model(input=[inputs_left,inputs_right], output=output)
    model.compile(loss='sparse_categorical_crossentropy', optimizer='Nadam', metrics=['accuracy'])
    model.fit([train_data1,train_data1], train_label1, batch_size=64 , epochs=30,verbose=2)
    a = np.argmax(model.predict([test_data1,test_data1]), axis=1)
    np.set_printoptions(threshold=100000000)
    # print("NLSTM_duo classification results:\n", a)
    BiLSTM_AC = accuracy_score(test_label1, a)
    BiLSTM_f1 = f1_score(test_label1, a, average='macro')
    print("NLSTM_duo_AC,NLSTM_duo_f1",BiLSTM_AC,BiLSTM_f1)
    return BiLSTM_AC, BiLSTM_f1
def NLSTM_attention():
    train_label1, train_data1, test_label1, test_data1 = load_data_old()
    print("NLSTM_attention Processing:")
    inputs_left = Input(shape=(num1,lenth),name='inp_left')
    # Conv_left = Conv1D(filters=32, kernel_size=5, padding='same', activation='relu')(inputs_left)
    lstm_out_left = NestedLSTM_True(64, depth=2, dropout=0,activation='relu')(inputs_left)
    a_left = Permute((2, 1), name='permute0_left')(lstm_out_left)
    a_left = Dense(num1, activation='relu', name='dense_left')(a_left)
    a_left_probs = Permute((2, 1), name='permute1_left')(a_left)
    output_attention_mul_left = merge.concatenate([lstm_out_left, a_left_probs], name='attention_mul_left')
    attention_mul_left = Flatten(name='flatten_left')(output_attention_mul_left)
    dp = Dropout(0, name='dp0')(attention_mul_left)
    output = Dense(8, activation='softmax')(dp)
    model = Model(input=inputs_left, output=output)
    model.compile(loss='sparse_categorical_crossentropy', optimizer='Nadam', metrics=['accuracy'])
    model.fit(train_data1, train_label1, batch_size=size_batch , epochs=30,verbose=2)
    a = np.argmax(model.predict(test_data1), axis=1)
    np.set_printoptions(threshold=100000000)
    # print("NLSTM_attention classification results:\n", a)
    BiLSTM_attention_AC = accuracy_score(test_label1, a)
    BiLSTM_attention_f1 = f1_score(test_label1, a, average='macro')
    print("NLSTM_attention_AC,NLSTM_attention_f1",BiLSTM_attention_AC,BiLSTM_attention_f1)
    return BiLSTM_attention_AC, BiLSTM_attention_f1

def NLSTM_duo_attention():
    train_label1, train_data1, test_label1, test_data1 = load_data_old()
    print("NLSTM_duo_attention Processing:")
    inputs_left = Input(shape=(num1,lenth),name='inp_left')
    conv_left = Conv1D(filters=32, kernel_size=5, padding='same', activation='relu')(inputs_left)
    inputs_right = Input(shape=(num1,lenth),name='inp_right')
    conv_inputs_right0 = Conv1D(filters=32, kernel_size=5, padding='same', activation='relu')(inputs_right)
    conv_inputs_right = Conv1D(filters=32, kernel_size=5, padding='same', activation='relu')(conv_inputs_right0)

    output_attention_mul_right = merge.concatenate([conv_left,conv_inputs_right])
    out0 = NestedLSTM_True(64, depth=2, dropout=0, recurrent_dropout=0.2,activation='relu')(output_attention_mul_right)
    attention_mul = attention_3d_block2(out0)
    # a_left = Permute((2, 1), name='permute0_left')(out0)
    # a_left = Dense(num1, activation='relu', name='dense_left')(a_left)
    # a_left_probs = Permute((2, 1), name='permute1_left')(a_left)
    # output_attention_mul_left = merge.average([out0, a_left_probs], name='attention_mul_left')
    attention_mul_left = Flatten(name='flatten_left')(attention_mul)
    # dp = Dropout(0.2, name='dp0')(attention_mul_left)
    output = Dense(8, activation='softmax')(attention_mul_left)
    model = Model(input=[inputs_left,inputs_right], output=output)
    model.compile(loss='sparse_categorical_crossentropy', optimizer='Nadam', metrics=['accuracy'])
    model.fit([train_data1,train_data1], train_label1, batch_size=64 , epochs=20,verbose=2)
    a = np.argmax(model.predict([test_data1,test_data1]), axis=1)
    # model.summary()
    np.set_printoptions(threshold=100000000)
    # print("NLSTM_duo_attention classification results:\n", a)
    BiLSTM_AC = accuracy_score(test_label1, a)
    BiLSTM_f1 = f1_score(test_label1, a, average='macro')
    print("NLSTM_duo_attention_AC,NLSTM_duo_attention_f1",BiLSTM_AC,BiLSTM_f1)
    return BiLSTM_AC, BiLSTM_f1

def NLSTM_duo1_attention():
    train_label1, train_data1, test_label1, test_data1 = load_data_old()
    print("NLSTM_duo1_attention Processing:")
    inputs_left = Input(shape=(num1,lenth),name='inp_left')
    conv_left = Conv1D(filters=32, kernel_size=5, padding='same', activation='relu')(inputs_left)
    # inputs_right = Input(shape=(num1,lenth),name='inp_right')
    # conv_inputs_right0 = Conv2D(filters=32, kernel_size=5, padding='same', activation='relu')(inputs_left)
    # a = MaxPool2D((2, 2), strides=(2, 2), name='block1_pool')(conv_inputs_right0)
    # conv_inputs_right1 = Reshape((-1, 1))(a)
    # conv_inputs_right = Conv1D(filters=32, kernel_size=5, padding='same', activation='relu')(conv_inputs_right0)
    #temp = Flatten()(conv_inputs_right0)
    # output_attention_mul_right = merge.concatenate([conv_left,conv_inputs_right])
    out0 = NestedLSTM_True(64, depth=2, dropout=0, recurrent_dropout=0.2,activation='relu')(conv_left)
    attention_mul = attention_3d_block2(out0)
    # a_left = Permute((2, 1), name='permute0_left')(out0)
    # a_left = Dense(num1, activation='relu', name='dense_left')(a_left)
    # a_left_probs = Permute((2, 1), name='permute1_left')(a_left)
    # output_attention_mul_left = merge.average([out0, a_left_probs], name='attention_mul_left')
    attention_mul_left = Flatten(name='flatten_left')(attention_mul)
    # dp = Dropout(0.2, name='dp0')(attention_mul_left)
    output = Dense(8, activation='softmax')(attention_mul_left)
    model = Model(input=[inputs_left], output=output)
    model.compile(loss='sparse_categorical_crossentropy', optimizer='Nadam', metrics=['accuracy'])
    model.fit([train_data1], train_label1, batch_size=64 , epochs=20,verbose=2)
    a = np.argmax(model.predict([test_data1]), axis=1)
    # model.summary()
    np.set_printoptions(threshold=100000000)
    # print("NLSTM_duo_attention classification results:\n", a)
    BiLSTM_AC = accuracy_score(test_label1, a)
    BiLSTM_f1 = f1_score(test_label1, a, average='macro')
    print("NLSTM_duo1_attention_AC,NLSTM_duo1_attention_f1",BiLSTM_AC,BiLSTM_f1)
    return BiLSTM_AC, BiLSTM_f1

def NLSTM_duo2_attention():
    train_label1, train_data1, test_label1, test_data1 = load_data_old()
    print("NLSTM_duo2_attention Processing:")
    inputs_left = Input(shape=(num1,lenth),name='inp_left')
    # conv_left = Conv1D(filters=32, kernel_size=5, padding='same', activation='relu')(inputs_left)
    # inputs_right = Input(shape=(num1,lenth),name='inp_right')
    conv_inputs_right0 = Conv1D(filters=32, kernel_size=5, padding='same', activation='relu')(inputs_left)
    conv_inputs_right = Conv1D(filters=32, kernel_size=5, padding='same', activation='relu')(conv_inputs_right0)

    # output_attention_mul_right = merge.concatenate([conv_left,conv_inputs_right])
    out0 = NestedLSTM_True(64, depth=2, dropout=0, recurrent_dropout=0.2,activation='relu')(conv_inputs_right)
    attention_mul = attention_3d_block2(out0)
    # a_left = Permute((2, 1), name='permute0_left')(out0)
    # a_left = Dense(num1, activation='relu', name='dense_left')(a_left)
    # a_left_probs = Permute((2, 1), name='permute1_left')(a_left)
    # output_attention_mul_left = merge.average([out0, a_left_probs], name='attention_mul_left')
    attention_mul_left = Flatten(name='flatten_left')(attention_mul)
    # dp = Dropout(0.2, name='dp0')(attention_mul_left)
    output = Dense(8, activation='softmax')(attention_mul_left)
    model = Model(input=[inputs_left], output=output)
    model.compile(loss='sparse_categorical_crossentropy', optimizer='Nadam', metrics=['accuracy'])
    model.fit([train_data1], train_label1, batch_size=64 , epochs=20,verbose=2)
    a = np.argmax(model.predict([test_data1]), axis=1)
    np.set_printoptions(threshold=100000000)
    # print("NLSTM_duo_attention classification results:\n", a)
    BiLSTM_AC = accuracy_score(test_label1, a)
    BiLSTM_f1 = f1_score(test_label1, a, average='macro')
    print("NLSTM_duo2_attention_AC,NLSTM_duo2_attention_f1",BiLSTM_AC,BiLSTM_f1)
    return BiLSTM_AC, BiLSTM_f1

global numbers
numbers = 5

# Accurary_LSTM =[]
# for i in range(numbers):
#     LSTM_AC,LSTM_f1 = Lstm_models()
#     Accurary_LSTM = np.append(Accurary_LSTM,LSTM_AC)
# print("Accurary_LSTM:",Accurary_LSTM)
# Average_LSTM = sum(Accurary_LSTM)/numbers
# print("Average_LSTM:",Average_LSTM)
#
# Accurary_GRU =[]
# for i in range(numbers):
#     GRU_AC,GRU_f1 = GRU_models()
#     Accurary_GRU = np.append(Accurary_GRU,GRU_AC)
# print("Accurary_GRU:",Accurary_GRU)
# Average_GRU = sum(Accurary_GRU)/numbers
# print("Average_GRU:",Average_GRU)
#
# Accurary_Bilstm =[]
# for i in range(numbers):
#     Bilstm_AC,Bilstm_f1 = Bilstm_models()
#     Accurary_Bilstm = np.append(Accurary_Bilstm,Bilstm_AC)
# print("Accurary_Bilstm:",Accurary_Bilstm)
# Average_Bilstm = sum(Accurary_Bilstm)/numbers
# print("Average_Bilstm:",Average_Bilstm)
#
# Accurary_Bilstm_duo =[]
# for i in range(numbers):
#     Bilstm_duo_AC,Bilstm_duo_f1 = BiLSTM_duo()
#     Accurary_Bilstm_duo = np.append(Accurary_Bilstm_duo,Bilstm_duo_AC)
# print("Accurary_Bilstm_duo:",Accurary_Bilstm_duo)
# Average_Bilstm_duo = sum(Accurary_Bilstm_duo)/numbers
# print("Average_Bilstm_duo:",Average_Bilstm_duo)
#
# Accurary_Bilstm_attention =[]
# for i in range(numbers):
#     BiLSTM_attention_AC,BiLSTM_attention_f1 = BiLSTM_attention()
#     Accurary_Bilstm_attention = np.append(Accurary_Bilstm_attention,BiLSTM_attention_AC)
# print("Accurary_Bilstm_attention:",Accurary_Bilstm_attention)
# Average_Bilstm_attention = sum(Accurary_Bilstm_attention)/numbers
# print("Average_Bilstm_attention:",Average_Bilstm_attention)
#
# Accurary_Nestlstm =[]
# for i in range(numbers):
#     Nestlstm_AC,Nestlstm_f1 = Nestlstm_models()
#     Accurary_Nestlstm = np.append(Accurary_Nestlstm,Nestlstm_AC)
# print("Accurary_Nestlstm:",Accurary_Nestlstm)
# Average_Nestlstm = sum(Accurary_Nestlstm)/numbers
# print("Average_Nestlstm:",Average_Nestlstm)
#
# Accurary_Nlstm_duo =[]
# for i in range(numbers):
#     Nlstm_duo_AC,Nlstm_duo_f1 = NLSTM_duo()
#     Accurary_Nlstm_duo = np.append(Accurary_Nlstm_duo,Nlstm_duo_AC)
# print("Accurary_Nlstm_duo:",Accurary_Nlstm_duo)
# Average_Nlstm_duo = sum(Accurary_Nlstm_duo)/numbers
# print("Average_Nlstm_duo:",Average_Nlstm_duo)
#
# Accurary_NLSTM_attention =[]
# for i in range(numbers):
#     NLSTM_attention_AC,NLSTM_attention_f1 = NLSTM_attention()
#     Accurary_NLSTM_attention = np.append(Accurary_NLSTM_attention,NLSTM_attention_AC)
# print("Accurary_NLSTM_attention:",Accurary_NLSTM_attention)
# Average_NLSTM_attention = sum(Accurary_NLSTM_attention)/numbers
# print("Average_NLSTM_attention:",Average_NLSTM_attention)

# Accurary_Nlstm_duo_attention =[]
# for i in range(numbers):
#     Nlstm_duo_AC,Nlstm_duo_f1 = NLSTM_duo_attention()
#     Accurary_Nlstm_duo_attention = np.append(Accurary_Nlstm_duo_attention,Nlstm_duo_AC)
# print("Accurary_Nlstm_duo_attention:",Accurary_Nlstm_duo_attention)
# Average_Nlstm_duo_attention = sum(Accurary_Nlstm_duo_attention)/numbers
# print("Average_Nlstm_duo_attention:",Average_Nlstm_duo_attention)

# Lstm_models()
# GRU_models()
# Bilstm_models()
# BiLSTM_duo()
# BiLSTM_attention()
# BiLSTM_duo_attention()
# Nestlstm_models()
# NLSTM_duo()
# NLSTM_attention()
for i in range(10):
    num1 = i+1
    NLSTM_duo_attention()
# NLSTM_duo_attention()
# NLSTM_duo1_attention()
# NLSTM_duo2_attention()

# print("Accurary_LSTM:",Accurary_LSTM)
# print("Average_LSTM:",Average_LSTM)
# print("Accurary_GRU:",Accurary_GRU)
# print("Average_GRU:",Average_GRU)
# print("Accurary_Bilstm:",Accurary_Bilstm)
# print("Average_Bilstm:",Average_Bilstm)
# print("Accurary_Bilstm_duo:",Accurary_Bilstm_duo)
# print("Average_Bilstm_duo:",Average_Bilstm_duo)
# print("Accurary_Bilstm_attention:",Accurary_Bilstm_attention)
# print("Average_Bilstm_attention:",Average_Bilstm_attention)
# print("Accurary_Nestlstm:",Accurary_Nestlstm)
# print("Average_Nestlstm:",Average_Nestlstm)
# print("Accurary_Nlstm_duo:",Accurary_Nlstm_duo)
# print("Average_Nlstm_duo:",Average_Nlstm_duo)
# print("Accurary_NLSTM_attention:",Accurary_NLSTM_attention)
# print("Average_NLSTM_attention:",Average_NLSTM_attention)
