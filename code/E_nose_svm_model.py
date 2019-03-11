#from sklearn import svm
import pickle
from sklearn import preprocessing

# 因为预测出来的值是数字，所以需要下面这两行进行标签转换
le = preprocessing.LabelEncoder()
le.fit(['background','banana','wine'])


# ”E_nose_svm_model.txt“这个文件需要与这个代码同目录
f = open('E_nose_svm_model.txt','rb')
s = f.read()
clf = pickle.loads(s)


# 传入的数据为二维的列表，可以同时测试多组数据，R1传感器的数据不要传入!!
test_data = [[10.3683, 10.4383, 11.6699, 13.4931, 13.3423, 8.04169, 8.73901, 26.2257, 59.0528]]
label = clf.predict(test_data)


# 打印预测结果
print(le.inverse_transform(label))
