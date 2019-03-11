import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
import statsmodels.api as sm
import pickle
from sklearn.svm import SVC

# 读取数据
dataset = pd.read_csv('data/HT_Sensor_dataset.dat', sep='  ', skiprows=1, header=None, engine='python')
dataset.columns = ['id', 'time', 'R1', 'R2', 'R3', 'R4', 'R5', 'R6', 'R7', 'R8', 'Temp.', 'Humidity']
dataset.set_index('id', inplace=True)

output = pd.read_csv('data/HT_Sensor_metadata.dat', skiprows=1, sep='\t', header=None)
output.columns = ['id', 'date', 'class', 't0', 'dt']

dataset = dataset.join(output, how='inner')
dataset.set_index(np.arange(dataset.shape[0]), inplace=True)
dataset['time'] += dataset['t0']
dataset.drop(['t0'], axis=1, inplace=True)

# 分析数据
xtrain, xtest, ytrain, ytest = train_test_split(
    dataset[[u'R1', u'R2', u'R3', u'R4', u'R5', u'R6', u'R7', u'R8', u'Temp.', u'Humidity']].values,
    dataset['class'].values, train_size=0.7)
for i in range(ytrain.shape[0]):
    if (ytrain[i] == 'background'):
        ytrain[i] = 0
    elif (ytrain[i] == 'banana'):
        ytrain[i] = 1
    else:
        ytrain[i] = 2

for i in range(ytest.shape[0]):
    if (ytest[i] == 'background'):
        ytest[i] = 0
    elif (ytest[i] == 'banana'):
        ytest[i] = 1
    else:
        ytest[i] = 2

ytrain = ytrain.astype('int64')
ytest = ytest.astype('int64')

# SVM算法（里面的C参数和gamma参数没调，多试几个理论上可以有更好的表现）
C_2d_range = [1e-2]
gamma_2d_range = [1e-1]
classifiers = []
for C in C_2d_range:
    for gamma in gamma_2d_range:
        clf = SVC(C=C, gamma=gamma)
        clf.fit(xtrain, ytrain)
        classifiers.append((C, gamma, clf))

# 保存训练模型
s = pickle.dumps(clf)
f = open('E_nose_svm_model.txt', 'wb')
f.write(s)