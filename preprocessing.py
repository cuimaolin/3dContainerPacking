from sklearn.model_selection import train_test_split
from sklearn import preprocessing

def data_preprocessing(data, test_size):
    # 分开input和lable
    data = data.values
    data_x = data[:, :-1]
    data_y = data[:, -1]
    # 预处理
    data_x = preprocessing.scale(data_x)    # x - u / σ
    data_y = data_y / 100   # 将百分比转换为小数 
    # 划分数据集，训练集：测试集 = 9 ：1
    return train_test_split(data_x, data_y, test_size=test_size, random_state=0)