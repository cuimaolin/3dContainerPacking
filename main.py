import pandas as pd
import argparse
from getFeature import getFeaure
from preprocessing import data_preprocessing
from sklearn.metrics import mean_squared_error
from math import sqrt
from model import linear_model, randomForest_model, network_model

# 简单参数设置
parser = argparse.ArgumentParser()
parser.add_argument('-p', '--test_size', default=0.1, type=float, help='how much percentage of the test set')
parser.add_argument('-m', '--model_name', default=0, type=int,\
                   help='what model you want to choose;\
                         0 for linear model;\
                         1 for random forest model;\
                         2 for neural network model')
args = parser.parse_args()

if __name__ == "__main__":
   '''
   得到特征
   '''
   data = getFeaure(ln_root='./data', ln_ult='./summary.xlsx')    # 原始数据
   data_afterdispose = getFeaure(ln_root='./data-afterdispose', ln_ult='./SCLPRes.xlsx')  # 合成数据
   data = pd.concat([data, data_afterdispose], axis=0)   # 将两种数据合并
   '''
   预处理
   '''
   train_x, test_x, train_y, test_y = data_preprocessing(data, test_size=args.test_size)
   '''
   训练模型并得到预测结果
   '''
   if args.model_name == 0:
      print('using linear model')
      pred_y = linear_model(train_x, test_x, train_y, test_y)
   if args.model_name == 1:
      print('using random forest model')
      pred_y = randomForest_model(train_x, test_x, train_y, test_y)
   if args.model_name == 2:
      print('using neural network model')
      pred_y = network_model(train_x, test_x, train_y, test_y)
   '''
   得到mse均方误差
   '''
   mse = sqrt(mean_squared_error(pred_y, test_y))
   print(mse)





