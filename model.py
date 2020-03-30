from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor

def linear_model(train_x, test_x, train_y, test_y):
    # 线性回归
    regr = LinearRegression()   # 括号内参数可调，具体可以百度
    regr.fit(train_x, train_y)  
    pred_y = regr.predict(test_x)

    return pred_y

def randomForest_model(train_x, test_x, train_y, test_y):
    # 随机森林
    rfr = RandomForestRegressor()   # 括号内参数可调，具体可以百度
    rfr.fit(train_x, train_y)
    pred_y = rfr.predict(test_x)

    return pred_y

def network_model(train_x, test_x, train_y, test_y):
    # 神经网络
    mlp = MLPRegressor(hidden_layer_sizes=(10,))     # 括号内参数可调，具体可以百度
    mlp.fit(train_x, train_y)
    pred_y = mlp.predict(test_x)

    return pred_y

# def lgb_model(train_x, test_x, train_y, test_y):
#     # TODO


