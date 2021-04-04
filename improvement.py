import numpy as np
import pandas as pd
from pandas import read_csv
global num
from sklearn import preprocessing
from nested_lstm import NestedLSTM
from sklearn.feature_selection import SelectKBest, chi2
from keras.models import *
from keras.layers import *
from sklearn.metrics import accuracy_score,f1_score
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.manifold import TSNE
from mpl_toolkits.mplot3d import Axes3D
from support_OL import ONLSTM
from sklearn.model_selection import GridSearchCV
import time
import tensorflow as tf
min_max_scaler = preprocessing.MinMaxScaler()
minmaxscaler = MinMaxScaler()
num =4
global num1
num1 =4
feature = 8
sum1=65
name = 'Level1'
def show():
    lev1_lab0 = pd.read_csv(r'data_set_1000train\\' + name + '_test.csv').values
    data = lev1_lab0[:, 1:]
    tsne = TSNE(n_components=3, init='pca', random_state=1)
    result = tsne.fit_transform(data)
    x_min, x_max = np.min(result), np.max(result)
    result = (result - x_min) / (x_max - x_min)# 这一步似乎让结果都变为0-1的数字
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(result[:100, 0], result[:100, 1], result[:100, 2], c='#00CED1', s=8, marker='o', label="normal")
    ax.scatter(result[100:200, 0], result[100:200, 1], result[100:200, 2], c='Chocolate', s=8, marker='o', label="CF")
    ax.scatter(result[200:300, 0], result[200:300, 1], result[200:300, 2], c='#DC143C', s=8, marker='o', label="EO")
    ax.scatter(result[300:400, 0], result[300:400, 1], result[300:400, 2], c='#A9A9A9', s=8, marker='o', label="NCR")
    ax.scatter(result[400:500, 0], result[400:500, 1], result[400:500, 2], c='#556B2F', s=8, marker='o', label="RCW")
    ax.scatter(result[500:600, 0], result[500:600, 1], result[500:600, 2], c='#9932CC', s=8, marker='o', label="REW")
    ax.scatter(result[600:700, 0], result[600:700, 1], result[600:700, 2], c='Gold', s=8, marker='o', label="RL")
    ax.scatter(result[700:800, 0], result[700:800, 1], result[700:800, 2], c='Indigo', s=8, marker='o', label="RO")
    ax.legend(loc=2)
    plt.title('Classification data visualization of ' + name)
    plt.show()

def plot_confusion_matrix(cm, labels_name, title):
    # cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]    # 归一化
    plt.imshow(cm,interpolation='nearest',)    # 在特定的窗口上显示图像
    plt.title(title)    # 图像标题
    plt.colorbar()
    num_local = np.array(range(len(labels_name)))
    plt.xticks(num_local, labels_name, rotation=90)    # 将标签印在x轴坐标上
    plt.yticks(num_local, labels_name)    # 将标签印在y轴坐标上
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def accuracy1(a,b):
    L1 = len(a)
    count =0
    # print(L1)
    for j in range(len(b)):
        if b[j]==a[j][0]:
            count=count+1
    return (count/L1)*100

def projection(real_data, predict_data,name1):
    L1 = len(real_data)
    L2 = len(predict_data)
    real_list = []
    real_list.append(real_data[:500])
    real_list.append(real_data[500:1000])
    real_list.append(real_data[1000:1500])
    real_list.append(real_data[1500:2000])
    real_list.append(real_data[2000:2500])
    real_list.append(real_data[2500:3000])
    real_list.append(real_data[3000:3500])
    real_list.append(real_data[3500:])
    # print(len(real_list))#8
    # print(real_list[6][40][0])#500
    # print(len(real_list[7]))#496
    predict_list = []
    predict_list.append(predict_data[:500])
    predict_list.append(predict_data[500:1000])
    predict_list.append(predict_data[1000:1500])
    predict_list.append(predict_data[1500:2000])
    predict_list.append(predict_data[2000:2500])
    predict_list.append(predict_data[2500:3000])
    predict_list.append(predict_data[3000:3500])
    predict_list.append(predict_data[3500:])
    # print(len(predict_list))#8
    # print(predict_list[6][40])#500
    # print(predict_list[7])#496

    error_list = []
    for i in range(len(predict_list)):##24
        accuracy2 = accuracy1(real_list[i], predict_list[i])
        error_list.append(accuracy2)
    error_list.append(error_list[0])
    print(len(error_list))##len(error_list)=9
    print(error_list)



    titles = np.arange(0, 8, 1).tolist()
    theta = np.arange(0, 2 * np.pi, (2/8) * np.pi)
    theta = theta.tolist()
    theta.append(0)
    #
    plt.figure(1, figsize=(5, 4))     # wide

    plt.rc('font', family='Times New Roman')
    ax1 = plt.subplot(projection='polar')#极坐标图
    ax1.set_thetagrids(np.arange(0.0, 360.0, 45.0), labels=titles, weight="bold", color="black", fontsize=16)
    ax1.set_rticks(np.arange(0, 100, 25))#(0, 100, 25)
    ax1.set_rlabel_position(0)
    ax1.set_rlim(0, 100)#(0, 100)

    ax1.set_theta_direction(-1)
    ax1.set_theta_zero_location('N')
    ax1.plot(theta, error_list, '--', linewidth=2.5, marker='o',color="black")
    plt.title(name1, fontsize=24, y=1.1)
    #
    plt.subplots_adjust(left=0.0, bottom=0.1, right=1.0, top=0.84)     # wide
    # plt.subplots_adjust(left=0.0, bottom=0.08, right=1.0, top=0.85)     # full
    #
    # # plt.savefig('result\\pics\\projection_' + str(name) + '_wide.png')
    # plt.savefig('C:\\Users\\dong\\Desktop\\improvement\\Polar\\fig_'+str(name1)+'_'+str(name)+'_'+str(feature)+'.png')
    #
    plt.show()
    # plt.close(1)

def feature_select():
    filename = r'data_set_5000train\\' + name + '_train.csv'
    # filename = "data_set_5000train\\Level1_train.csv"
    dataFame = read_csv(filename)
    column_headers = list(dataFame.columns.values)
    select = dataFame.values
    train1 = select[0:4000, :]
    test1 = select[4000:5000, :]
    train2 = select[5000:9000, :]
    test2 = select[9000:10000, :]
    train3 = select[10000:14000, :]
    test3 = select[14000:15000, :]
    train4 = select[15000:19000, :]
    test4 = select[19000:20000, :]
    train5 = select[20000:24000, :]
    test5 = select[24000:25000, :]
    train6 = select[25000:29000, :]
    test6 = select[29000:30000, :]
    train7 = select[30000:34000, :]
    test7 = select[34000:35000, :]
    train8 = select[35000:39000, :]
    test8 = select[39000:40000, :]

    train = np.concatenate((train1, train2, train3, train4, train5, train6, train7, train8,), axis=0)
    test = np.concatenate((test1, test2, test3, test4, test5, test6, test7, test8,), axis=0)
    trainx = train[:,1:]
    testx = test[:,1:]
    trainx = minmaxscaler.fit_transform(trainx)
    testx = minmaxscaler.fit_transform(testx)
    trainy = train[:,0]
    testy = test[:,0]
    trainy = trainy.reshape(trainy.shape[0],1)
    aa = SelectKBest(chi2, k=8)
    X_new =aa.fit_transform(trainx, trainy)
    # X_new =SelectKBest(lambda X, Y: array(map(lambda x: pearsonr(x, Y), X.T)).T, k=2).fit_transform(trainx, trainy)
    # print(trainx.shape)
    # print(X_new.shape)
    # print("aa.scores_:",aa.scores_)
    aa.scores_[53] = 0
    # print("11111111111",len(aa.scores_))#65
    # print("11111111111", type(aa.scores_))
    # print("aa.scores_:", aa.scores_)
    # print("aa.scores_:", sum(aa.scores_))
    for i in range(len(aa.scores_)):
        aa.scores_[i] = aa.scores_[i]/sum(aa.scores_);
    aa.scores_ = aa.scores_.tolist()
    print(aa.scores_)
    a = sorted(aa.scores_,reverse=True)
    print("分数从大到小排列为：",a)
    order=[]
    for i in range(len(aa.scores_)):
        order.append(aa.scores_.index(a[i])-1)
    print(order)
    features = []
    for i in range(len(order)):
        features.append(column_headers[order[i]+1])
    print("选择的特征是:",features)

    plt.bar(range(len(aa.scores_)), aa.scores_)
    plt.show()

    # trainx = train[:, [56, 62, 55, 57, 27, 58, 45, 52, 32, 60]]
    trainx = train[:, [6, 4, 48, 8, 18, 2, 28, 46, 30, 61, 5, 25, 32, 26, 9, 33]]
    #[6, 4, 48, 8, 18, 2, 28, 46, 30, 61, 5, 25, 32, 26, 9, 33]
    pcc = np.corrcoef(trainx.T) * 0.5 + 0.5
    print(pcc)
    labels_name = ['1', '2', '3', '4', '5', '6', '7', '8','9', '10', '11', '12', '13', '14', '15', '16']
    plot_confusion_matrix(pcc, labels_name, "Confusion Matrix")
    # plt.savefig('/HAR_cm.png', format='png')
    plt.show()

    # indices = np.argsort(aa.scores_)[::-1]
    # k_best_features = list(trainx.tolist.columns.values[indices[0:8]])
    # print('k best features are: ', k_best_features)

    return None

def load_ML():
    np.random.seed(2)
    lev1_lab0 = pd.read_csv(r'data_set0\\level1\\lev1_lab0.csv').values
    lev1_lab1 = pd.read_csv(r'data_set0\\level1\\lev1_lab1.csv').values
    lev1_lab2 = pd.read_csv(r'data_set0\\level1\\lev1_lab2.csv').values
    lev1_lab3 = pd.read_csv(r'data_set0\\level1\\lev1_lab3.csv').values
    lev1_lab4 = pd.read_csv(r'data_set0\\level1\\lev1_lab4.csv').values
    lev1_lab5 = pd.read_csv(r'data_set0\\level1\\lev1_lab5.csv').values
    lev1_lab6 = pd.read_csv(r'data_set0\\level1\\lev1_lab6.csv').values
    lev1_lab7 = pd.read_csv(r'data_set0\\level1\\lev1_lab7.csv').values

    lev1_lab0_train = lev1_lab0[0:4000, :]
    lev1_lab0_test = lev1_lab0[4000:4500, :]

    lev1_lab1_train = lev1_lab1[0:4000, :]
    lev1_lab1_test = lev1_lab1[4000:4500, :]

    lev1_lab2_train = lev1_lab2[0:4000, :]
    lev1_lab2_test = lev1_lab2[4000:4500, :]

    lev1_lab3_train = lev1_lab3[0:4000, :]
    lev1_lab3_test = lev1_lab3[4000:4500, :]

    lev1_lab4_train = lev1_lab4[0:4000, :]
    lev1_lab4_test = lev1_lab4[4000:4500, :]

    lev1_lab5_train = lev1_lab5[0:4000, :]
    lev1_lab5_test = lev1_lab5[4000:4500, :]

    lev1_lab6_train = lev1_lab6[0:4000, :]
    lev1_lab6_test = lev1_lab6[4000:4500, :]

    lev1_lab7_train = lev1_lab7[0:4000, :]
    lev1_lab7_test = lev1_lab7[4000:4500, :]

    # "#合并训练集测试集\n",
    lev1_train = np.concatenate([lev1_lab0_train, lev1_lab1_train, lev1_lab2_train, lev1_lab3_train,
                            lev1_lab4_train, lev1_lab5_train, lev1_lab6_train, lev1_lab7_train], axis=0)

    lev1_test = np.concatenate([lev1_lab0_test, lev1_lab1_test, lev1_lab2_test, lev1_lab3_test,
                           lev1_lab4_test, lev1_lab5_test, lev1_lab6_test, lev1_lab7_test], axis=0)

    # 故障等级2中，读取原始数据集\n",
    lev2_lab0 = pd.read_csv(r'data_set0\\level2\\lev2_lab0.csv').values
    lev2_lab1 = pd.read_csv(r'data_set0\\level2\\lev2_lab1.csv').values
    lev2_lab2 = pd.read_csv(r'data_set0\\level2\\lev2_lab2.csv').values
    lev2_lab3 = pd.read_csv(r'data_set0\\level2\\lev2_lab3.csv').values
    lev2_lab4 = pd.read_csv(r'data_set0\\level2\\lev2_lab4.csv').values
    lev2_lab5 = pd.read_csv(r'data_set0\\level2\\lev2_lab5.csv').values
    lev2_lab6 = pd.read_csv(r'data_set0\\level2\\lev2_lab6.csv').values
    lev2_lab7 = pd.read_csv(r'data_set0\\level2\\lev2_lab7.csv').values

    # 将5000个数据划分为训练集、验证集、测试集\n",
    lev2_lab0_train = lev2_lab0[0:4000, :]
    lev2_lab0_test = lev2_lab0[4000:4500, :]

    lev2_lab1_train = lev2_lab1[0:4000, :]
    lev2_lab1_test = lev2_lab1[4000:4500, :]

    lev2_lab2_train = lev2_lab2[0:4000, :]
    lev2_lab2_test = lev2_lab2[4000:4500, :]

    lev2_lab3_train = lev2_lab3[0:4000, :]
    lev2_lab3_test = lev2_lab3[4000:4500, :]

    lev2_lab4_train = lev2_lab4[0:4000, :]
    lev2_lab4_test = lev2_lab4[4000:4500, :]

    lev2_lab5_train = lev2_lab5[0:4000, :]
    lev2_lab5_test = lev2_lab5[4000:4500, :]

    lev2_lab6_train = lev2_lab6[0:4000, :]
    lev2_lab6_test = lev2_lab6[4000:4500, :]

    lev2_lab7_train = lev2_lab7[0:4000, :]
    lev2_lab7_test = lev2_lab7[4000:4500, :]

    # 合并训练集测试集\n",
    lev2_train = np.concatenate([lev2_lab0_train, lev2_lab1_train, lev2_lab2_train, lev2_lab3_train,
                            lev2_lab4_train, lev2_lab5_train, lev2_lab6_train, lev2_lab7_train], axis=0)

    lev2_test = np.concatenate([lev2_lab0_test, lev2_lab1_test, lev2_lab2_test, lev2_lab3_test,
                           lev2_lab4_test, lev2_lab5_test, lev2_lab6_test, lev2_lab7_test], axis=0)

    # 故障等级3中，读取原始数据集\n",
    lev3_lab0 = pd.read_csv(r'data_set0\\level3\\lev3_lab0.csv').values
    lev3_lab1 = pd.read_csv(r'data_set0\\level3\\lev3_lab1.csv').values
    lev3_lab2 = pd.read_csv(r'data_set0\\level3\\lev3_lab2.csv').values
    lev3_lab3 = pd.read_csv(r'data_set0\\level3\\lev3_lab3.csv').values
    lev3_lab4 = pd.read_csv(r'data_set0\\level3\\lev3_lab4.csv').values
    lev3_lab5 = pd.read_csv(r'data_set0\\level3\\lev3_lab5.csv').values
    lev3_lab6 = pd.read_csv(r'data_set0\\level3\\lev3_lab6.csv').values
    lev3_lab7 = pd.read_csv(r'data_set0\\level3\\lev3_lab7.csv').values

    # 讲5000个数据划分为训练集、验证集、测试集\n",
    lev3_lab0_train = lev3_lab0[0:4000, :]
    lev3_lab0_test = lev3_lab0[4000:4500, :]

    lev3_lab1_train = lev3_lab1[0:4000, :]
    lev3_lab1_test = lev3_lab1[4000:4500, :]

    lev3_lab2_train = lev3_lab2[0:4000, :]
    lev3_lab2_test = lev3_lab2[4000:4500, :]

    lev3_lab3_train = lev3_lab3[0:4000, :]
    lev3_lab3_test = lev3_lab3[4000:4500, :]

    lev3_lab4_train = lev3_lab4[0:4000, :]
    lev3_lab4_test = lev3_lab4[4000:4500, :]

    lev3_lab5_train = lev3_lab5[0:4000, :]
    lev3_lab5_test = lev3_lab5[4000:4500, :]

    lev3_lab6_train = lev3_lab6[0:4000, :]
    lev3_lab6_test = lev3_lab6[4000:4500, :]

    lev3_lab7_train = lev3_lab7[0:4000, :]
    lev3_lab7_test = lev3_lab7[4000:4500, :]
    # 合并训练集测试集\n",
    lev3_train = np.concatenate([lev3_lab0_train, lev3_lab1_train, lev3_lab2_train, lev3_lab3_train,
                            lev3_lab4_train, lev3_lab5_train, lev3_lab6_train, lev3_lab7_train], axis=0)

    lev3_test = np.concatenate([lev3_lab0_test, lev3_lab1_test, lev3_lab2_test, lev3_lab3_test,
                           lev3_lab4_test, lev3_lab5_test, lev3_lab6_test, lev3_lab7_test], axis=0)

    # "#故障等级4中，读取原始数据集\n",
    lev4_lab0 = pd.read_csv(r'data_set0\\level4\\lev4_lab0.csv').values
    lev4_lab1 = pd.read_csv(r'data_set0\\level4\\lev4_lab1.csv').values
    lev4_lab2 = pd.read_csv(r'data_set0\\level4\\lev4_lab2.csv').values
    lev4_lab3 = pd.read_csv(r'data_set0\\level4\\lev4_lab3.csv').values
    lev4_lab4 = pd.read_csv(r'data_set0\\level4\\lev4_lab4.csv').values
    lev4_lab5 = pd.read_csv(r'data_set0\\level4\\lev4_lab5.csv').values
    lev4_lab6 = pd.read_csv(r'data_set0\\level4\\lev4_lab6.csv').values
    lev4_lab7 = pd.read_csv(r'data_set0\\level4\\lev4_lab7.csv').values

    # 讲5000个数据划分为训练集、验证集、测试集\n",
    lev4_lab0_train = lev4_lab0[0:4000, :]
    lev4_lab0_test = lev4_lab0[4000:4500, :]

    lev4_lab1_train = lev4_lab1[0:4000, :]
    lev4_lab1_test = lev4_lab1[4000:4500, :]

    lev4_lab2_train = lev4_lab2[0:4000, :]
    lev4_lab2_test = lev4_lab2[4000:4500, :]

    lev4_lab3_train = lev4_lab3[0:4000, :]
    lev4_lab3_test = lev4_lab3[4000:4500, :]

    lev4_lab4_train = lev4_lab4[0:4000, :]
    lev4_lab4_test = lev4_lab4[4000:4500, :]

    lev4_lab5_train = lev4_lab5[0:4000, :]
    lev4_lab5_test = lev4_lab5[4000:4500, :]

    lev4_lab6_train = lev4_lab6[0:4000, :]
    lev4_lab6_test = lev4_lab6[4000:4500, :]

    lev4_lab7_train = lev4_lab7[0:4000, :]
    lev4_lab7_test = lev4_lab7[4000:4500, :]

    # 合并训练集测试集\n",
    lev4_train = np.concatenate([lev4_lab0_train, lev4_lab1_train, lev4_lab2_train, lev4_lab3_train,
                            lev4_lab4_train, lev4_lab5_train, lev4_lab6_train, lev4_lab7_train], axis=0)


    lev4_test = np.concatenate([lev4_lab0_test, lev4_lab1_test, lev4_lab2_test, lev4_lab3_test,
                           lev4_lab4_test, lev4_lab5_test, lev4_lab6_test, lev4_lab7_test], axis=0)

    return lev1_train,lev1_test,lev2_train,lev2_test,lev3_train,lev3_test,lev4_train,lev4_test

def load_data_det_8(train_data,test_data):

    # train_X = train_data[['FWC', 'FWE', 'TCA', 'Tolerance%', 'PO_feed', 'PO_net', 'TWCD', 'VE']]
    # test_X = test_data[['FWC', 'FWE', 'TCA', 'Tolerance%', 'PO_feed', 'PO_net', 'TWCD', 'VE']]
    # train_X = train_data[['TCI', 'TEO', 'PO_feed', 'TCO','Evap Tons', 'TEI', 'TCA', 'TO_feed']]
    # test_X = test_data[['TCI', 'TEO', 'PO_feed', 'TCO','Evap Tons', 'TEI', 'TCA', 'TO_feed']]

    #[6, 4, 48, 8, 18, 2, 28, 46, 30, 61, 5, 25, 32, 26, 9, 33]
    # train_X = train_data[:, [6, 4,  48, 8, 18, 2]]
    # test_X = test_data[:, [6, 4, 48, 8, 18, 2]]
    # train_X = train_data[:, [6, 4, 48, 8, 18, 2, 28, 46]]
    # test_X = test_data[:, [ 6, 4,48, 8, 18, 2, 28, 46]]
    train_X = train_data[:, [6, 4,  48, 8, 18, 2, 28, 46, 30, 61]]
    test_X = test_data[:, [6, 4, 48, 8, 18, 2, 28, 46, 30, 61]]

    # [25, 26, 29, 43, 48, 49, 50, 57]
    # [6, 4, 48, 8, 18, 2, 28, 46]



    train_X = train_X

    test_X = test_X

    train_X = train_X.reshape(train_X.shape[0],train_X.shape[1],1)
    test_X = test_X.reshape(test_X.shape[0],test_X.shape[1],1)
    train_Y = train_data[:,0]
    test_Y = test_data[:,0]

    train_Y = np.array(train_Y)
    test_Y = np.array(test_Y)
    train_Y = train_Y.reshape(-1,1)
    test_Y = test_Y.reshape(-1,1)
    return train_X,train_Y,test_X,test_Y

def load_DL():
    filename_train = r'data_set_5000train\\' + name + '_train.csv'
    # filename_train = "data_set\Level2_train.csv"
    dataSet_train = pd.read_csv(filename_train, encoding='gbk')

    filename_test = r'data_set_5000train\\' + name + '_train.csv'
    # filename_test = "data_set\Level2_test.csv"
    dataSet_test = pd.read_csv(filename_test, encoding='gbk')

    dataSet_train = dataSet_train.values
    dataSet_test = dataSet_test.values
    #
    train1 = dataSet_train[0:4000, :]
    test1 = dataSet_test[4000:4500, :]
    train2 = dataSet_train[5000:9000, :]
    test2 = dataSet_test[9000:9500, :]
    train3 = dataSet_train[10000:14000, :]
    test3 = dataSet_test[14000:14500, :]
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

    true_fault_data_train1 = np.concatenate((train1, train2, train3, train4, train5, train6, train7, train8))
    true_fault_data_test1 = np.concatenate((test1, test2, test3, test4, test5, test6, test7, test8))
    print(true_fault_data_train1.shape)
    print(true_fault_data_test1.shape)
    #[6, 4, 48, 8, 18, 2, 28, 46, 30, 61, 5, 25, 32, 26, 9, 33]
    # true_fault_data_train = true_fault_data_train1[:, [6,4,48,8,18,2,28,46,30,61,5,25,32,26,9,33]]
    # true_fault_data_train = true_fault_data_train1[:, [56, 62, 55, 57, 27, 58, 45, 52]]
    # true_fault_data_train = true_fault_data_train1[:, [6, 4, 48, 8, 18, 2, 28, 46, 30, 61]]
    true_fault_data_train = true_fault_data_train1[:, [6, 4, 48, 8, 18, 2, 28, 46]]
    # true_fault_data_train = true_fault_data_train1[:, [6, 4, 48, 8, 18, 2]]
    # true_fault_data_train = dataSet_train[:, 1:]
    true_fault_label_train = true_fault_data_train1[:, 0]

    # true_fault_data_test = true_fault_data_test1[:, [6,4,48,8,18,2,28,46,30,61,5,25,32,26,9,33]]
    # true_fault_data_test = true_fault_data_test1[:,[56, 62, 55, 57, 27, 58, 45, 52]]
    # true_fault_data_test = true_fault_data_test1[:,[6, 4, 48, 8, 18, 2, 28, 46, 30, 61]]
    true_fault_data_test = true_fault_data_test1[:,[6, 4, 48, 8, 18, 2, 28, 46]]
    # true_fault_data_test = true_fault_data_test1[:, [6, 4, 48, 8, 18, 2]]
    # true_fault_data_test = dataSet_test[:, 1:]
    true_fault_label_test = true_fault_data_test1[:, 0]
    # 归一化
    true_fault_data_train = min_max_scaler.fit_transform(true_fault_data_train)
    true_fault_data_test = min_max_scaler.transform(true_fault_data_test)
    # print("true_fault_data_train的大小:",true_fault_data_train.shape)
    # print("true_fault_data_test的大小:",true_fault_data_test.shape)

    train_data1 = np.array(true_fault_data_train)
    train_label1 = np.array(true_fault_label_train)
    test_data1 = np.array(true_fault_data_test)
    test_label1 = np.array(true_fault_label_test)
    train_label1 = train_label1.reshape(32000, 1)
    test_label1 = test_label1.reshape(4000, 1)

    #######################################################################################

    trX = train_data1[:, :]
    trY = train_label1[num1:, :]

    teX = test_data1[:, :]
    teY = test_label1[num1:, :]

    trainX = []

    for i in range(trX.shape[0] - num1):
        tempX = trX[i:i + num1, :]
        # print(type(tempX))
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

def ML():
    # start_time = time.time()#参数调优要花费1小时之久~~
    lev1_train, lev1_test, lev2_train, lev2_test, lev3_train, lev3_test, lev4_train, lev4_test = load_ML()
    train_X_det1, train_Y_det1, test_X_det1, test_Y_det1 = load_data_det_8(lev4_train,lev4_test)
    print('train_X_det1.shape', train_X_det1.shape)
    print('train_Y_det1.shape:', train_Y_det1.shape)
    print('test_X_det1.shape', test_X_det1.shape)
    print('test_Y_det1.shape:', test_Y_det1.shape)
    train_X_det1 = train_X_det1.reshape(train_X_det1.shape[0],train_X_det1.shape[1])
    test_X_det1 = test_X_det1.reshape(test_X_det1.shape[0],test_X_det1.shape[1])
    # svc_level1 = SVC(kernel='sigmoid', C=9, gamma=0.0001, decision_function_shape='ovr')
    # svc_level1 = SVC()
    # # 网格搜索法#要花费很长时间
    # # parameters = {'kernel': ('linear', 'rbf'), 'C': [1, 5,10,20,100,200,500],'gamma':[0.01,0.001,0.0001]}
    # # grid_search = GridSearchCV(svc_level1, parameters)
    # # grid_result = grid_search.fit(train_X_det1, train_Y_det1.ravel())
    # # print("Best: %f using %s"%(grid_result.best_score_,grid_search.best_params_))
    # # # sorted(clf.cv_results_.keys())
    # # # print(sorted(clf.cv_results_.keys()))
    # # end_time = time.time()
    # # print("time:%d" % (end_time - start_time))
    #
    # svc_level1.set_params(kernel='rbf',C = 5,gamma = 0.001,probability=True).fit(train_X_det1, train_Y_det1.ravel())#到达最优
    # a_svc_det1 = svc_level1.predict(test_X_det1)
    # np.set_printoptions(threshold=100000000)
    # print("a_svc_det1:", a_svc_det1)
    # svc_AC_det1 = accuracy_score(test_Y_det1, a_svc_det1)
    # print('svc_AC_det1=', svc_AC_det1)



    # classifier = LogisticRegression(random_state=5,solver='sag',C=10,)
    # classifier.fit(train_X_det1, train_Y_det1.ravel())
    # lg_predict = classifier.predict(test_X_det1)
    # np.set_printoptions(threshold=100000000)
    # # print("lg_predict:", lg_predict)
    # LG_AC = accuracy_score(test_Y_det1, lg_predict)
    # LG_f1 = f1_score(test_Y_det1, lg_predict, average='macro')
    # print("LG_AC:", LG_AC)

    # dtc = DecisionTreeClassifier(criterion="gini",max_features=8,min_samples_split=8)
    # dtc.fit(train_X_det1, train_Y_det1.ravel())
    # dt_pre = dtc.predict(test_X_det1)
    # np.set_printoptions(threshold=100000000)
    # # print("dt_pre:", dt_pre)
    # DT_AC = accuracy_score(test_Y_det1, dt_pre)
    # DT_f1 = f1_score(test_Y_det1, dt_pre, average='macro')
    # print("DT_AC:", DT_AC)

    rfc1 = RandomForestClassifier(n_estimators=100, max_features=4, random_state=2)  # 随机森林分类器
    rfc1.fit(train_X_det1, train_Y_det1.ravel())
    RF_pre = rfc1.predict(test_X_det1)
    np.set_printoptions(threshold=100000000)
    # print("RF_pre:", RF_pre)
    RF_AC = accuracy_score(test_Y_det1, RF_pre)
    RF_f1 = f1_score(test_Y_det1, RF_pre, average='macro')
    name = 'RF'
    print("RF_AC:", RF_AC)
    _ = projection(test_Y_det1, RF_pre, name)
    # knn = KNeighborsClassifier(weights='distance',p=2,leaf_size=50)
    # knn.fit(train_X_det1, train_Y_det1.ravel())
    # knn_predict = knn.predict(test_X_det1)
    # np.set_printoptions(threshold=100000000)
    # # print("knn_predict:", knn_predict)
    # KNN_AC = accuracy_score(test_Y_det1, knn_predict)
    # KNN_f1 = f1_score(test_Y_det1, knn_predict, average='macro')
    # print("KNN_AC:", KNN_AC)

def Lstm_models():
    train_label1,train_data1,test_label1,test_data1 = load_DL()

    print("LSTM Processing:")
    model = Sequential()
    model.add(LSTM(units=64, input_shape=(num1, feature), return_sequences=True, activation='relu'))
    model.add(LSTM(units=64, return_sequences=False, activation='relu'))
    model.add(Dropout(0.2))
    # model.add(LSTM(units = 64,return_sequences=False,activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(8, activation='softmax'))
    model.compile(loss='sparse_categorical_crossentropy', optimizer='Nadam',
                  metrics=['sparse_categorical_accuracy'])  # accuracy
    history = model.fit(train_data1, train_label1, batch_size=64, epochs=30, verbose=2,
                        validation_data=(test_data1, test_label1), validation_freq=1)
    a = np.argmax(model.predict(test_data1), axis=1)
    acc = history.history['sparse_categorical_accuracy']
    val_acc = history.history['val_sparse_categorical_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    font2 = {'size': 12}
    fig = plt.figure(figsize=(9,4))

    plt.subplot(1, 2, 1)
    plt.plot(acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.xlabel("Epoch",font2)
    plt.ylabel("Accuracy",font2)
    plt.legend()

    # plt.subplot(1, 2, 2)
    ax = fig.add_subplot(1, 2, 2)
    ax.plot(loss, label='Training Loss')
    ax.plot(val_loss, label='Validation Loss')
    ax.set_xlabel("Epoch",font2)
    ax.set_ylabel("Loss",font2)
    # ax.yaxis.set_ticks_position('right')
    # ax.yaxis.set_label_position('right')
    plt.legend()
    plt.suptitle('Accuracy and loss of training set and verification set of LSTM')
    plt.show()

    name = "LSTM"
    LSTM_AC = accuracy_score(test_label1, a)
    LSTM_f1 = f1_score(test_label1, a, average='macro')
    print("LSTM_AC,LSTM_f1", LSTM_AC, LSTM_f1)
    # _ = projection(test_label1, a,name)
    return LSTM_AC, LSTM_f1

def GRU_models():
    train_label1,train_data1,test_label1,test_data1 = load_DL()
    print("train_label1.shape:", train_label1.shape)
    print("train_data1.shape:", train_data1.shape)
    print("test_label1.shape:", test_label1.shape)
    print("test_data1.shape:", test_data1.shape)
    model = Sequential()
    model.add(GRU(units = 64,input_shape=(num1,feature),return_sequences=True,activation='relu'))
    model.add(GRU(units = 64,return_sequences=False,activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(8,activation='softmax'))
    model.compile(loss='sparse_categorical_crossentropy', optimizer='Nadam',
                  metrics=['sparse_categorical_accuracy'])  # accuracy
    history = model.fit(train_data1, train_label1, batch_size=64, epochs=30, verbose=2,
                        validation_data=(test_data1, test_label1), validation_freq=1)
    a = np.argmax(model.predict(test_data1), axis=1)
    acc = history.history['sparse_categorical_accuracy']
    val_acc = history.history['val_sparse_categorical_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    # font2 = {'size': 12}
    # fig = plt.figure(figsize=(9, 4))
    #
    # plt.subplot(1, 2, 1)
    # plt.plot(acc, label='Training Accuracy')
    # plt.plot(val_acc, label='Validation Accuracy')
    # plt.xlabel("Epoch", font2)
    # plt.ylabel("Accuracy", font2)
    # plt.legend()
    #
    # # plt.subplot(1, 2, 2)
    # ax = fig.add_subplot(1, 2, 2)
    # ax.plot(loss, label='Training Loss')
    # ax.plot(val_loss, label='Validation Loss')
    # ax.set_xlabel("Epoch", font2)
    # ax.set_ylabel("Loss", font2)
    # # ax.yaxis.set_ticks_position('right')
    # # ax.yaxis.set_label_position('right')
    # plt.legend()
    #
    # plt.suptitle('Accuracy and loss of training set and verification set of GRU')
    # plt.show()
    # print("GRU classification results:\n",a)
    name = "GRU"
    GRU_AC = accuracy_score(test_label1, a)
    GRU_f1 = f1_score(test_label1, a, average='macro')
    print("GRU_AC,GRU_f1",GRU_AC,GRU_f1)
    _ = projection(test_label1, a,name)
    return GRU_AC,GRU_f1

def Slstm_models():
    train_label1, train_data1, test_label1, test_data1 = load_DL()

    print("SLSTM Processing:")
    model = Sequential()
    model.add(LSTM(units=128, input_shape=(num1, feature), return_sequences=True, activation='relu'))
    model.add(LSTM(units=64, return_sequences=True, activation='relu'))
    model.add(LSTM(units=64, return_sequences=True, activation='relu'))
    model.add(LSTM(units=64, return_sequences=False, activation='relu'))
    model.add(Dropout(0.2))
    # model.add(LSTM(units = 64,return_sequences=False,activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(8, activation='softmax'))
    model.compile(loss='sparse_categorical_crossentropy', optimizer='Nadam',
                  metrics=['sparse_categorical_accuracy'])  # accuracy
    history = model.fit(train_data1, train_label1, batch_size=64, epochs=30, verbose=2,
                        validation_data=(test_data1, test_label1), validation_freq=1)
    a = np.argmax(model.predict(test_data1), axis=1)
    acc = history.history['sparse_categorical_accuracy']
    val_acc = history.history['val_sparse_categorical_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    font2 = {'size': 12}
    fig = plt.figure(figsize=(9, 4))

    plt.subplot(1, 2, 1)
    plt.plot(acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.xlabel("Epoch", font2)
    plt.ylabel("Accuracy", font2)
    plt.legend()

    # plt.subplot(1, 2, 2)
    ax = fig.add_subplot(1, 2, 2)
    ax.plot(loss, label='Training Loss')
    ax.plot(val_loss, label='Validation Loss')
    ax.set_xlabel("Epoch", font2)
    ax.set_ylabel("Loss", font2)
    # ax.yaxis.set_ticks_position('right')
    # ax.yaxis.set_label_position('right')
    plt.legend()

    plt.suptitle('Accuracy and loss of training set and verification set of SLSTM')
    # plt.show()
    name = "SLSTM"
    LSTM_AC = accuracy_score(test_label1, a)
    LSTM_f1 = f1_score(test_label1, a, average='macro')
    print("SLSTM_AC,SLSTM_f1", LSTM_AC, LSTM_f1)
    # _ = projection(test_label1, a, name)
    return model

def Bilstm_models():
    train_label1,train_data1,test_label1,test_data1 = load_DL()
    print("Bilstm Processing:")
    model1 = Sequential()
    # model1.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu', input_shape=(num1,lenth)))
    # model1.add(Bidirectional(LSTM(units=64, return_sequences=False, activation='relu')))
    model1.add(Bidirectional(LSTM(units=64, input_shape=(num1, feature), return_sequences=False, activation='relu')))
    model1.add(Dense(8,activation='softmax'))
    model1.compile(loss='sparse_categorical_crossentropy', optimizer='Nadam',
                  metrics=['sparse_categorical_accuracy'])  # accuracy
    history = model1.fit(train_data1, train_label1, batch_size=64, epochs=30, verbose=2,
                        validation_data=(test_data1, test_label1), validation_freq=1)
    a = np.argmax(model1.predict(test_data1), axis=1)
    acc = history.history['sparse_categorical_accuracy']
    val_acc = history.history['val_sparse_categorical_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    font2 = {'size': 12}
    # fig = plt.figure(figsize=(9,4))
    #
    # plt.subplot(1, 2, 1)
    # plt.plot(acc, label='Training Accuracy')
    # plt.plot(val_acc, label='Validation Accuracy')
    # plt.xlabel("Epoch",font2)
    # plt.ylabel("Accuracy",font2)
    # plt.legend()
    #
    # # plt.subplot(1, 2, 2)
    # ax = fig.add_subplot(1, 2, 2)
    # ax.plot(loss, label='Training Loss')
    # ax.plot(val_loss, label='Validation Loss')
    # ax.set_xlabel("Epoch",font2)
    # ax.set_ylabel("Loss",font2)
    # # ax.yaxis.set_ticks_position('right')
    # # ax.yaxis.set_label_position('right')
    # plt.legend()
    #
    # plt.suptitle('Accuracy and loss of training set and verification set of BiLSTM')
    # plt.show()
    a = np.argmax(model1.predict(test_data1),axis=1)
    np.set_printoptions(threshold=100000000)
    # print("BiLSTM classification results:\n",a)
    name = "BiLSTM"
    BiLSTM_AC = accuracy_score(test_label1, a)
    BiLSTM_f1 = f1_score(test_label1, a, average='macro')
    print("BiLSTM_AC,BiLSTM_f1",BiLSTM_AC,BiLSTM_f1)
    _ = projection(test_label1, a,name)
    return BiLSTM_AC,BiLSTM_f1

def Bigru_models():
    train_label1,train_data1,test_label1,test_data1 = load_DL()
    print("BiGRU Processing:")
    model1 = Sequential()
    # model1.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu', input_shape=(num1,lenth)))
    # model1.add(Bidirectional(LSTM(units=64, return_sequences=False, activation='relu')))
    model1.add(Bidirectional(GRU(units=64, input_shape=(num1, feature), return_sequences=False, activation='relu')))
    model1.add(Dense(64, activation='relu'))
    model1.add(Dense(8,activation='softmax'))
    model1.compile(loss='sparse_categorical_crossentropy', optimizer='Nadam',
                  metrics=['sparse_categorical_accuracy'])  # accuracy
    history = model1.fit(train_data1, train_label1, batch_size=64, epochs=30, verbose=2,
                        validation_data=(test_data1, test_label1), validation_freq=1)
    a = np.argmax(model1.predict(test_data1), axis=1)
    acc = history.history['sparse_categorical_accuracy']
    val_acc = history.history['val_sparse_categorical_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    # font2 = {'size': 12}
    # fig = plt.figure(figsize=(9,4))
    #
    # plt.subplot(1, 2, 1)
    # plt.plot(acc, label='Training Accuracy')
    # plt.plot(val_acc, label='Validation Accuracy')
    # plt.xlabel("Epoch",font2)
    # plt.ylabel("Accuracy",font2)
    # plt.legend()
    #
    # # plt.subplot(1, 2, 2)
    # ax = fig.add_subplot(1, 2, 2)
    # ax.plot(loss, label='Training Loss')
    # ax.plot(val_loss, label='Validation Loss')
    # ax.set_xlabel("Epoch",font2)
    # ax.set_ylabel("Loss",font2)
    # # ax.yaxis.set_ticks_position('right')
    # # ax.yaxis.set_label_position('right')
    # plt.legend()
    #
    # plt.suptitle('Accuracy and loss of training set and verification set of BiGRU')
    # # plt.show()
    a = np.argmax(model1.predict(test_data1),axis=1)
    np.set_printoptions(threshold=100000000)
    # print("BiLSTM classification results:\n",a)
    name = "BiGRU"
    BiLSTM_AC = accuracy_score(test_label1, a)
    BiLSTM_f1 = f1_score(test_label1, a, average='macro')
    print("BiGRU_AC,BiGRU_f1",BiLSTM_AC,BiLSTM_f1)
    _ = projection(test_label1, a,name)
    return BiLSTM_AC,BiLSTM_f1

def Nestlstm_models():
    print("NLSTM Processing:")
    callback1 = [tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=100, restore_best_weights=True)]
    train_label1,train_data1,test_label1,test_data1 = load_DL()
    model1 = Sequential()
    # model1.add(NestedLSTM(64, depth=2, dropout=0, recurrent_dropout=0.2,activation='relu',return_sequences=True))
    # model1.add(NestedLSTM(64, depth=2, dropout=0, recurrent_dropout=0,activation='relu',return_sequences=True))
    # model1.add(NestedLSTM(64, depth=2, dropout=0, recurrent_dropout=0,activation='relu',return_sequences=True))
    model1.add(NestedLSTM(64, depth=2, dropout=0, recurrent_dropout=0.2,activation='relu'))
    model1.add(Dense(64, activation='relu'))
    model1.add(Dense(8,activation='softmax'))
    model1.compile(loss='sparse_categorical_crossentropy', optimizer='Nadam',
                  metrics=['sparse_categorical_accuracy'])  # accuracy
    history = model1.fit(train_data1, train_label1, batch_size=64, epochs=300, verbose=2,
                        validation_data=(test_data1, test_label1), validation_freq=1,callbacks = callback1)
    a = np.argmax(model1.predict(test_data1), axis=1)
    acc = history.history['sparse_categorical_accuracy']
    val_acc = history.history['val_sparse_categorical_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    #
    # font2 = {'size': 12}
    # fig = plt.figure(figsize=(9, 4))
    #
    # plt.subplot(1, 2, 1)
    # plt.plot(acc, label='Training Accuracy')
    # plt.plot(val_acc, label='Validation Accuracy')
    # plt.xlabel("Epoch", font2)
    # plt.ylabel("Accuracy", font2)
    # plt.legend()
    #
    # # plt.subplot(1, 2, 2)
    # ax = fig.add_subplot(1, 2, 2)
    # ax.plot(loss, label='Training Loss')
    # ax.plot(val_loss, label='Validation Loss')
    # ax.set_xlabel("Epoch", font2)
    # ax.set_ylabel("Loss", font2)
    # # ax.yaxis.set_ticks_position('right')
    # # ax.yaxis.set_label_position('right')
    # plt.legend()
    #
    # plt.suptitle('Accuracy and loss of training set and verification set of NLSTM')
    # plt.show()
    a = np.argmax(model1.predict(test_data1),axis=1)
    # print("Initial data classification results:\n",test_label1)
    # print("NLSTM classification results:\n",a)
    name = "NLSTM"
    _ = projection(test_label1, a,name)
    NLSTM_AC = accuracy_score(test_label1, a)
    NLSTM_f1 = f1_score(test_label1, a, average='macro')
    print("NLSTM_AC,NLSTM_f1",NLSTM_AC,NLSTM_f1)
    return NLSTM_AC,NLSTM_f1

def Binlstm_models():
    train_label1,train_data1,test_label1,test_data1 = load_DL()
    print("BiNlstm Processing:")
    model1 = Sequential()
    # model1.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu', input_shape=(num1,lenth)))
    # model1.add(Bidirectional(LSTM(units=64, return_sequences=False, activation='relu')))
    model1.add(Bidirectional(NestedLSTM(64, depth=2, dropout=0, recurrent_dropout=0.2,activation='relu')))
    model1.add(Dense(8,activation='softmax'))
    model1.compile(loss='sparse_categorical_crossentropy', optimizer='Nadam',
                  metrics=['sparse_categorical_accuracy'])  # accuracy
    history = model1.fit(train_data1, train_label1, batch_size=64, epochs=30, verbose=2,
                        validation_data=(test_data1, test_label1), validation_freq=1)
    a = np.argmax(model1.predict(test_data1), axis=1)
    acc = history.history['sparse_categorical_accuracy']
    val_acc = history.history['val_sparse_categorical_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    font2 = {'size': 12}
    fig = plt.figure(figsize=(9,4))

    plt.subplot(1, 2, 1)
    plt.plot(acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.xlabel("Epoch",font2)
    plt.ylabel("Accuracy",font2)
    plt.legend()

    # plt.subplot(1, 2, 2)
    ax = fig.add_subplot(1, 2, 2)
    ax.plot(loss, label='Training Loss')
    ax.plot(val_loss, label='Validation Loss')
    ax.set_xlabel("Epoch",font2)
    ax.set_ylabel("Loss",font2)
    # ax.yaxis.set_ticks_position('right')
    # ax.yaxis.set_label_position('right')
    plt.legend()

    plt.suptitle('Accuracy and loss of training set and verification set of BiLSTM')
    # plt.show()
    a = np.argmax(model1.predict(test_data1),axis=1)
    np.set_printoptions(threshold=100000000)
    # print("BiLSTM classification results:\n",a)
    name = "BiLSTM"
    BiLSTM_AC = accuracy_score(test_label1, a)
    BiLSTM_f1 = f1_score(test_label1, a, average='macro')
    print("BiNLSTM_AC,BiNLSTM_f1",BiLSTM_AC,BiLSTM_f1)
    # _ = projection(test_label1, a,name)
    return BiLSTM_AC,BiLSTM_f1

def main():
    # _ = show()
    # _ = load_DL()
    # _ = load_ML()
    _ = feature_select()
    # _ = ML()
    # _ = Lstm_models()
    # _ = GRU_models()
    # _ = Slstm_models()
    # _ = Bilstm_models()
    # _ = Bigru_models()
    # _ = Nestlstm_models()
    # _ = Binlstm_models()
    return None

if __name__ == '__main__':
    _ = main()