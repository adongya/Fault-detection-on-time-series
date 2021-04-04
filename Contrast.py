import pandas as pd
import numpy as np
import tensorflow as tf
import numpy as np
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR,SVC,LinearSVR,LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model.logistic import LogisticRegression
min_max_scaler = MinMaxScaler()
np.random.seed(2)
lev1_lab0 = pd.read_csv(r'data_set0\\level1\\lev1_lab0.csv')
lev1_lab1 = pd.read_csv(r'data_set0\\level1\\lev1_lab1.csv')
lev1_lab2 = pd.read_csv(r'data_set0\\level1\\lev1_lab2.csv')
lev1_lab3 = pd.read_csv(r'data_set0\\level1\\lev1_lab3.csv')
lev1_lab4 = pd.read_csv(r'data_set0\\level1\\lev1_lab4.csv')
lev1_lab5 = pd.read_csv(r'data_set0\\level1\\lev1_lab5.csv')
lev1_lab6 = pd.read_csv(r'data_set0\\level1\\lev1_lab6.csv')
lev1_lab7 = pd.read_csv(r'data_set0\\level1\\lev1_lab7.csv')

# "#丢弃包含NAN的数据\n",
lev1_lab0 = lev1_lab0.dropna()
lev1_lab1 = lev1_lab1.dropna()
lev1_lab2 = lev1_lab2.dropna()
lev1_lab3 = lev1_lab3.dropna()
lev1_lab4 = lev1_lab4.dropna()
lev1_lab5 = lev1_lab5.dropna()
lev1_lab6 = lev1_lab6.dropna()
lev1_lab7 = lev1_lab7.dropna()

# "#从原始数据集中随即抽取5000个数据\n",
# lev1_lab0 = lev1_lab0.sample(n=5000)
# lev1_lab1 = lev1_lab1.sample(n=5000)
# lev1_lab2 = lev1_lab2.sample(n=5000)
# lev1_lab3 = lev1_lab3.sample(n=5000)
# lev1_lab4 = lev1_lab4.sample(n=5000)
# lev1_lab5 = lev1_lab5.sample(n=5000)
# lev1_lab6 = lev1_lab6.sample(n=5000)
# lev1_lab7 = lev1_lab7.sample(n=5000)

# "#讲5000个数据划分为训练集、验证集、测试集(5:2:3)\n",
lev1_lab0_train = lev1_lab0.iloc[0:2501,:]
lev1_lab0_val   = lev1_lab0.iloc[2501:4901,:]
lev1_lab0_test  = lev1_lab0.iloc[4901:5000,:]

lev1_lab1_train = lev1_lab1.iloc[0:2501,:]
lev1_lab1_val   = lev1_lab1.iloc[2501:4901,:]
lev1_lab1_test  = lev1_lab1.iloc[4901:5000,:]

lev1_lab2_train = lev1_lab2.iloc[0:2501,:]
lev1_lab2_val   = lev1_lab2.iloc[2501:4901,:]
lev1_lab2_test  = lev1_lab2.iloc[4901:5000,:]

lev1_lab3_train = lev1_lab3.iloc[0:2501,:]
lev1_lab3_val   = lev1_lab3.iloc[2501:4901,:]
lev1_lab3_test  = lev1_lab3.iloc[4901:5000,:]

lev1_lab4_train = lev1_lab4.iloc[0:2501,:]
lev1_lab4_val   = lev1_lab4.iloc[2501:4901,:]
lev1_lab4_test  = lev1_lab4.iloc[4901:5000,:]

lev1_lab5_train = lev1_lab5.iloc[0:2501,:]
lev1_lab5_val   = lev1_lab5.iloc[2501:4901,:]
lev1_lab5_test  = lev1_lab5.iloc[4901:5000,:]

lev1_lab6_train = lev1_lab6.iloc[0:2501,:]
lev1_lab6_val   = lev1_lab6.iloc[2501:4901,:]
lev1_lab6_test  = lev1_lab6.iloc[4901:5000,:]

lev1_lab7_train = lev1_lab7.iloc[0:2501,:]
lev1_lab7_val   = lev1_lab7.iloc[2501:4901,:]
lev1_lab7_test  = lev1_lab7.iloc[4901:5000,:]

# "#合并训练集测试集\n",
lev1_train =  pd.concat([lev1_lab0_train,lev1_lab1_train,lev1_lab2_train,lev1_lab3_train,
                        lev1_lab4_train,lev1_lab5_train,lev1_lab6_train,lev1_lab7_train],axis=0)

lev1_val  =  pd.concat([lev1_lab0_val,lev1_lab1_val,lev1_lab2_val,lev1_lab3_val,lev1_lab4_val,
                       lev1_lab5_val,lev1_lab6_val,lev1_lab7_val],axis=0)

lev1_test = pd.concat([lev1_lab0_test,lev1_lab1_test,lev1_lab2_test,lev1_lab3_test,
                        lev1_lab4_test,lev1_lab5_test,lev1_lab6_test,lev1_lab7_test],axis=0)

#故障等级2中，读取原始数据集\n",
lev2_lab0 = pd.read_csv(r'data_set0\\level2\\lev2_lab0.csv')
lev2_lab1 = pd.read_csv(r'data_set0\\level2\\lev2_lab1.csv')
lev2_lab2 = pd.read_csv(r'data_set0\\level2\\lev2_lab2.csv')
lev2_lab3 = pd.read_csv(r'data_set0\\level2\\lev2_lab3.csv')
lev2_lab4 = pd.read_csv(r'data_set0\\level2\\lev2_lab4.csv')
lev2_lab5 = pd.read_csv(r'data_set0\\level2\\lev2_lab5.csv')
lev2_lab6 = pd.read_csv(r'data_set0\\level2\\lev2_lab6.csv')
lev2_lab7 = pd.read_csv(r'data_set0\\level2\\lev2_lab7.csv')

#丢弃包含NAN的数据\n",
lev2_lab0 = lev2_lab0.dropna()
lev2_lab1 = lev2_lab1.dropna()
lev2_lab2 = lev2_lab2.dropna()
lev2_lab3 = lev2_lab3.dropna()
lev2_lab4 = lev2_lab4.dropna()
lev2_lab5 = lev2_lab5.dropna()
lev2_lab6 = lev2_lab6.dropna()
lev2_lab7 = lev2_lab7.dropna()

#从原始数据集中随即抽取5000个数据\n",
# lev2_lab0 = lev2_lab0.sample(n=5000)
# lev2_lab1 = lev2_lab1.sample(n=5000)
# lev2_lab2 = lev2_lab2.sample(n=5000)
# lev2_lab3 = lev2_lab3.sample(n=5000)
# lev2_lab4 = lev2_lab4.sample(n=5000)
# lev2_lab5 = lev2_lab5.sample(n=5000)
# lev2_lab6 = lev2_lab6.sample(n=5000)
# lev2_lab7 = lev2_lab7.sample(n=5000)

#将5000个数据划分为训练集、验证集、测试集\n",
lev2_lab0_train = lev2_lab0.iloc[0:2501,:]
lev2_lab0_val   = lev2_lab0.iloc[2501:4901,:]
lev2_lab0_test  = lev2_lab0.iloc[4901:5000,:]

lev2_lab1_train = lev2_lab1.iloc[0:2501,:]
lev2_lab1_val   = lev2_lab1.iloc[2501:4901,:]
lev2_lab1_test  = lev2_lab1.iloc[4901:5000,:]

lev2_lab2_train = lev2_lab2.iloc[0:2501,:]
lev2_lab2_val   = lev2_lab2.iloc[2501:4901,:]
lev2_lab2_test  = lev2_lab2.iloc[4901:5000,:]

lev2_lab3_train = lev2_lab3.iloc[0:2501,:]
lev2_lab3_val   = lev2_lab3.iloc[2501:4901,:]
lev2_lab3_test  = lev2_lab3.iloc[4901:5000,:]

lev2_lab4_train = lev2_lab4.iloc[0:2501,:]
lev2_lab4_val   = lev2_lab4.iloc[2501:4901,:]
lev2_lab4_test  = lev2_lab4.iloc[4901:5000,:]

lev2_lab5_train = lev2_lab5.iloc[0:2501,:]
lev2_lab5_val   = lev2_lab5.iloc[2501:4901,:]
lev2_lab5_test  = lev2_lab5.iloc[4901:5000,:]

lev2_lab6_train = lev2_lab6.iloc[0:2501,:]
lev2_lab6_val   = lev2_lab6.iloc[2501:4901,:]
lev2_lab6_test  = lev2_lab6.iloc[4901:5000,:]

lev2_lab7_train = lev2_lab7.iloc[0:2501,:]
lev2_lab7_val   = lev2_lab7.iloc[2501:4901,:]
lev2_lab7_test  = lev2_lab7.iloc[4901:5000,:]

#合并训练集测试集\n",
lev2_train =  pd.concat([lev2_lab0_train,lev2_lab1_train,lev2_lab2_train,lev2_lab3_train,
                        lev2_lab4_train,lev2_lab5_train,lev2_lab6_train,lev2_lab7_train],axis=0)

lev2_val =  pd.concat([lev2_lab0_val,lev2_lab1_val,lev2_lab2_val,lev2_lab3_val,lev2_lab4_val,
                       lev2_lab5_val,lev2_lab6_val,lev2_lab7_val],axis=0)

lev2_test = pd.concat([lev2_lab0_test,lev2_lab1_test,lev2_lab2_test,lev2_lab3_test,
                        lev2_lab4_test,lev2_lab5_test,lev2_lab6_test,lev2_lab7_test],axis=0)

#故障等级3中，读取原始数据集\n",
lev3_lab0 = pd.read_csv(r'data_set0\\level3\\lev3_lab0.csv')
lev3_lab1 = pd.read_csv(r'data_set0\\level3\\lev3_lab1.csv')
lev3_lab2 = pd.read_csv(r'data_set0\\level3\\lev3_lab2.csv')
lev3_lab3 = pd.read_csv(r'data_set0\\level3\\lev3_lab3.csv')
lev3_lab4 = pd.read_csv(r'data_set0\\level3\\lev3_lab4.csv')
lev3_lab5 = pd.read_csv(r'data_set0\\level3\\lev3_lab5.csv')
lev3_lab6 = pd.read_csv(r'data_set0\\level3\\lev3_lab6.csv')
lev3_lab7 = pd.read_csv(r'data_set0\\level3\\lev3_lab7.csv')

#丢弃包含NAN的数据\n",
lev3_lab0 = lev3_lab0.dropna()
lev3_lab1 = lev3_lab1.dropna()
lev3_lab2 = lev3_lab2.dropna()
lev3_lab3 = lev3_lab3.dropna()
lev3_lab4 = lev3_lab4.dropna()
lev3_lab5 = lev3_lab5.dropna()
lev3_lab6 = lev3_lab6.dropna()
lev3_lab7 = lev3_lab7.dropna()

#从原始数据集中随即抽取1200个数据\n",
# lev3_lab0 = lev3_lab0.sample(n=5000)
# lev3_lab1 = lev3_lab1.sample(n=5000)
# lev3_lab2 = lev3_lab2.sample(n=5000)
# lev3_lab3 = lev3_lab3.sample(n=5000)
# lev3_lab4 = lev3_lab4.sample(n=5000)
# lev3_lab5 = lev3_lab5.sample(n=5000)
# lev3_lab6 = lev3_lab6.sample(n=5000)
# lev3_lab7 = lev3_lab7.sample(n=5000)

#讲5000个数据划分为训练集、验证集、测试集\n",
lev3_lab0_train = lev3_lab0.iloc[0:2501,:]
lev3_lab0_val   = lev3_lab0.iloc[2501:4901,:]
lev3_lab0_test  = lev3_lab0.iloc[4901:5000,:]

lev3_lab1_train = lev3_lab1.iloc[0:2501,:]
lev3_lab1_val   = lev3_lab1.iloc[2501:4901,:]
lev3_lab1_test  = lev3_lab1.iloc[4901:5000,:]

lev3_lab2_train = lev3_lab2.iloc[0:2501,:]
lev3_lab2_val   = lev3_lab2.iloc[2501:4901,:]
lev3_lab2_test  = lev3_lab2.iloc[4901:5000,:]

lev3_lab3_train = lev3_lab3.iloc[0:2501,:]
lev3_lab3_val   = lev3_lab3.iloc[2501:4901,:]
lev3_lab3_test  = lev3_lab3.iloc[4901:5000,:]

lev3_lab4_train = lev3_lab4.iloc[0:2501,:]
lev3_lab4_val   = lev3_lab4.iloc[2501:4901,:]
lev3_lab4_test  = lev3_lab4.iloc[4901:5000,:]

lev3_lab5_train = lev3_lab5.iloc[0:2501,:]
lev3_lab5_val   = lev3_lab5.iloc[2501:4901,:]
lev3_lab5_test  = lev3_lab5.iloc[4901:5000,:]

lev3_lab6_train = lev3_lab6.iloc[0:2501,:]
lev3_lab6_val   = lev3_lab6.iloc[2501:4901,:]
lev3_lab6_test  = lev3_lab6.iloc[4901:5000,:]

lev3_lab7_train = lev3_lab7.iloc[0:2501,:]
lev3_lab7_val   = lev3_lab7.iloc[2501:4901,:]
lev3_lab7_test  = lev3_lab7.iloc[4901:5000,:]
#合并训练集测试集\n",
lev3_train =  pd.concat([lev3_lab0_train,lev3_lab1_train,lev3_lab2_train,lev3_lab3_train,
                        lev3_lab4_train,lev3_lab5_train,lev3_lab6_train,lev3_lab7_train],axis=0)

lev3_val =  pd.concat([lev3_lab0_val,lev3_lab1_val,lev3_lab2_val,lev3_lab3_val,lev3_lab4_val,
                       lev3_lab5_val,lev3_lab6_val,lev3_lab7_val],axis=0)

lev3_test = pd.concat([lev3_lab0_test,lev3_lab1_test,lev3_lab2_test,lev3_lab3_test,
                        lev3_lab4_test,lev3_lab5_test,lev3_lab6_test,lev3_lab7_test],axis=0)


# "#故障等级4中，读取原始数据集\n",
lev4_lab0 = pd.read_csv(r'data_set0\\level4\\lev4_lab0.csv')
lev4_lab1 = pd.read_csv(r'data_set0\\level4\\lev4_lab1.csv')
lev4_lab2 = pd.read_csv(r'data_set0\\level4\\lev4_lab2.csv')
lev4_lab3 = pd.read_csv(r'data_set0\\level4\\lev4_lab3.csv')
lev4_lab4 = pd.read_csv(r'data_set0\\level4\\lev4_lab4.csv')
lev4_lab5 = pd.read_csv(r'data_set0\\level4\\lev4_lab5.csv')
lev4_lab6 = pd.read_csv(r'data_set0\\level4\\lev4_lab6.csv')
lev4_lab7 = pd.read_csv(r'data_set0\\level4\\lev4_lab7.csv')

#丢弃包含NAN的数据\n",
lev4_lab0 = lev4_lab0.dropna()
lev4_lab1 = lev4_lab1.dropna()
lev4_lab2 = lev4_lab2.dropna()
lev4_lab3 = lev4_lab3.dropna()
lev4_lab4 = lev4_lab4.dropna()
lev4_lab5 = lev4_lab5.dropna()
lev4_lab6 = lev4_lab6.dropna()
lev4_lab7 = lev4_lab7.dropna()

#从原始数据集中随即抽取5000个数据\n",
# lev4_lab0 = lev4_lab0.sample(n=5000)
# lev4_lab1 = lev4_lab1.sample(n=5000)
# lev4_lab2 = lev4_lab2.sample(n=5000)
# lev4_lab3 = lev4_lab3.sample(n=5000)
# lev4_lab4 = lev4_lab4.sample(n=5000)
# lev4_lab5 = lev4_lab5.sample(n=5000)
# lev4_lab6 = lev4_lab6.sample(n=5000)
# lev4_lab7 = lev4_lab7.sample(n=5000)

#讲5000个数据划分为训练集、验证集、测试集\n",
lev4_lab0_train = lev4_lab0.iloc[0:2501,:]
lev4_lab0_val   = lev4_lab0.iloc[2501:4901,:]
lev4_lab0_test  = lev4_lab0.iloc[4901:5000,:]

lev4_lab1_train = lev4_lab1.iloc[0:2501,:]
lev4_lab1_val   = lev4_lab1.iloc[2501:4901,:]
lev4_lab1_test  = lev4_lab1.iloc[4901:5000,:]

lev4_lab2_train = lev4_lab2.iloc[0:2501,:]
lev4_lab2_val   = lev4_lab2.iloc[2501:4901,:]
lev4_lab2_test  = lev4_lab2.iloc[4901:5000,:]

lev4_lab3_train = lev4_lab3.iloc[0:2501,:]
lev4_lab3_val   = lev4_lab3.iloc[2501:4901,:]
lev4_lab3_test  = lev4_lab3.iloc[4901:5000,:]

lev4_lab4_train = lev4_lab4.iloc[0:2501,:]
lev4_lab4_val   = lev4_lab4.iloc[2501:4901,:]
lev4_lab4_test  = lev4_lab4.iloc[4901:5000,:]

lev4_lab5_train = lev4_lab5.iloc[0:2501,:]
lev4_lab5_val   = lev4_lab5.iloc[2501:4901,:]
lev4_lab5_test  = lev4_lab5.iloc[4901:5000,:]

lev4_lab6_train = lev4_lab6.iloc[0:2501,:]
lev4_lab6_val   = lev4_lab6.iloc[2501:4901,:]
lev4_lab6_test  = lev4_lab6.iloc[4901:5000,:]

lev4_lab7_train = lev4_lab7.iloc[0:2501,:]
lev4_lab7_val   = lev4_lab7.iloc[2501:4901,:]
lev4_lab7_test  = lev4_lab7.iloc[4901:5000,:]

#合并训练集测试集\n",
lev4_train =  pd.concat([lev4_lab0_train,lev4_lab1_train,lev4_lab2_train,lev4_lab3_train,
                        lev4_lab4_train,lev4_lab5_train,lev4_lab6_train,lev4_lab7_train],axis=0)

lev4_val =  pd.concat([lev4_lab0_val,lev4_lab1_val,lev4_lab2_val,lev4_lab3_val,lev4_lab4_val,
                       lev4_lab5_val,lev4_lab6_val,lev4_lab7_val],axis=0)

lev4_test = pd.concat([lev4_lab0_test,lev4_lab1_test,lev4_lab2_test,lev4_lab3_test,
                        lev4_lab4_test,lev4_lab5_test,lev4_lab6_test,lev4_lab7_test],axis=0)

def load_data_det_8(train_data,val_data,test_data):
    # [25,26,28,43,48,49,50,57]
    train_X = train_data[['FWC', 'FWE', 'TCA', 'Tolerance%', 'PO_feed', 'PO_net', 'TWCD', 'VE']]
    val_X = val_data[['FWC', 'FWE', 'TCA', 'Tolerance%', 'PO_feed', 'PO_net', 'TWCD', 'VE']]
    test_X = test_data[['FWC', 'FWE', 'TCA', 'Tolerance%', 'PO_feed', 'PO_net', 'TWCD', 'VE']]
    #
    # train_X = train_data[['TCI', 'TEO', 'PO_feed', 'TCO','Evap Tons', 'TEI', 'TCA', 'TO_feed']]
    # val_X = val_data[['TCI', 'TEO', 'PO_feed', 'TCO','Evap Tons', 'TEI', 'TCA', 'TO_feed']]
    # test_X = test_data[['TCI', 'TEO', 'PO_feed', 'TCO','Evap Tons', 'TEI', 'TCA', 'TO_feed']]


    train_X = train_X.values
    val_X = val_X.values
    test_X = test_X.values

    train_X = train_X.reshape(train_X.shape[0],train_X.shape[1],1)
    val_X = val_X.reshape(val_X.shape[0],val_X.shape[1],1)
    test_X = test_X.reshape(test_X.shape[0],test_X.shape[1],1)
    train_Y = train_data.iloc[:,0]
    val_Y = val_data.iloc[:,0]
    test_Y = test_data.iloc[:,0]

    train_Y = np.array(train_Y)
    val_Y = np.array(val_Y)
    test_Y = np.array(test_Y)
    train_Y = train_Y.reshape(-1,1)
    val_Y = val_Y.reshape(-1,1)
    test_Y = test_Y.reshape(-1,1)

    return train_X,train_Y,val_X,val_Y,test_X,test_Y

train_X_det1,train_Y_det1,val_X_det1,val_Y_det1,test_X_det1,test_Y_det1 = load_data_det_8(lev1_train,lev1_val,lev1_test)
print('train_X_det1.shape',train_X_det1.shape)
print('train_Y_det1.shape:',train_Y_det1.shape)
print('val_X_det1.shape',val_X_det1.shape)
print('val_Y_det1.shape:',val_Y_det1.shape)
print('test_X_det1.shape',test_X_det1.shape)
print('test_Y_det1.shape:',test_Y_det1.shape)

train_X_det1_svc = train_X_det1.reshape(train_X_det1.shape[0],train_X_det1.shape[1])
test_X_det1_svc = test_X_det1.reshape(test_X_det1.shape[0],test_X_det1.shape[1])

svc_level1= SVC(kernel='sigmoid', C=9, gamma=0.01,decision_function_shape='ovr')
svc_level1.set_params(kernel='rbf', probability=True).fit(train_X_det1_svc, train_Y_det1.ravel())
a_svc_det1 = svc_level1.predict(test_X_det1_svc)
np.set_printoptions(threshold=100000000)
print("a_svc_det1:",a_svc_det1)
svc_AC_det1 = accuracy_score(test_Y_det1, a_svc_det1)
print('svc_AC_det1=',svc_AC_det1)

classifier = LogisticRegression()
classifier.fit(train_X_det1_svc, train_Y_det1.ravel())
lg_predict = classifier.predict(test_X_det1_svc)
np.set_printoptions(threshold=100000000)
print("lg_predict:", lg_predict)
LG_AC = accuracy_score(test_Y_det1, lg_predict)
LG_f1 = f1_score(test_Y_det1, lg_predict, average='macro')
print("LG_AC:", LG_AC)

dtc = DecisionTreeClassifier()
dtc.fit(train_X_det1_svc, train_Y_det1.ravel())
dt_pre = dtc.predict(test_X_det1_svc)
np.set_printoptions(threshold=100000000)
print("dt_pre:", dt_pre)
DT_AC = accuracy_score(test_Y_det1, dt_pre)
DT_f1 = f1_score(test_Y_det1, dt_pre, average='macro')
print("DT_AC:", DT_AC)

rfc1 = RandomForestClassifier(n_estimators=40, max_depth=None, min_samples_split=2, random_state=2)  # 随机森林分类器
rfc1.fit(train_X_det1_svc, train_Y_det1.ravel())
RF_pre = rfc1.predict(test_X_det1_svc)
np.set_printoptions(threshold=100000000)
print("RF_pre:", RF_pre)
RF_AC = accuracy_score(test_Y_det1, RF_pre)
RF_f1 = f1_score(test_Y_det1, RF_pre, average='macro')
print("RF_AC:", RF_AC)

AdaBoostModel = AdaBoostClassifier(base_estimator=None, n_estimators=50, learning_rate=1, algorithm='SAMME.R',
                                   random_state=None)
AdaBoostModel.fit(train_X_det1_svc, train_Y_det1.ravel())
AdaBoost_pre = AdaBoostModel.predict(test_X_det1_svc)
np.set_printoptions(threshold=100000000)
print("AdaBoost_pre:", AdaBoost_pre)
AdaBoost_AC = accuracy_score(test_Y_det1, AdaBoost_pre)
AdaBoost_f1 = f1_score(test_Y_det1, AdaBoost_pre, average='macro')
print("AdaBoost_AC:", AdaBoost_AC)

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(train_X_det1_svc, train_Y_det1.ravel())
knn_predict = knn.predict(test_X_det1_svc)
np.set_printoptions(threshold=100000000)
print("knn_predict:", knn_predict)
KNN_AC = accuracy_score(test_Y_det1, knn_predict)
KNN_f1 = f1_score(test_Y_det1, knn_predict, average='macro')
print("KNN_AC:", KNN_AC)