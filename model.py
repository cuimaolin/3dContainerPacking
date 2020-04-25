from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
import lightgbm as lgb

def linear_model(train_x, train_y, test_x):
    # 线性回归
    regr = LinearRegression()
    regr.fit(train_x, train_y) 
    pred_y = regr.predict(test_x)

    return pred_y

def randomForest_model(train_x, train_y, test_x):
    # 随机森林
    rfr = RandomForestRegressor(n_estimators=100, oob_score=True)
    rfr.fit(train_x, train_y)
    pred_y = rfr.predict(test_x)

    return pred_y

def network_model(train_x, train_y, test_x):
    # 神经网络
    mlp = MLPRegressor(hidden_layer_sizes=(100,))
    mlp.fit(train_x, train_y)
    pred_y = mlp.predict(test_x)

    return pred_y

def lightgbm_model(train_x, train_y, test_x):
    # lightgbm
    lightgbm = lgb.LGBMRegressor()
    lightgbm.fit(train_x, train_y)
    pred_y = lightgbm.predict(test_x)

    return pred_y
