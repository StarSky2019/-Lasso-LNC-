# import packages
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os
from sklearn.svm import SVR
from sklearn.metrics import r2_score  # R square
from sklearn.metrics import mean_squared_error  # 均方误差
from sklearn.metrics import mean_absolute_error  # 平方绝对误差
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
import warnings

warnings.filterwarnings('ignore')


# 数据预处理
def Data_preprocessing(df):
    # 数据归一化
    mn_X = MinMaxScaler()
    features = [col for col in df.columns if col != 'Y']
    df[features] = mn_X.fit_transform(df[features])

    # df = pd.get_dummies(df)

    return df


# 模型结果评估
def evaluate_model(y_predict, y_test):
    # R2评价指标
    score = r2_score(y_test, y_predict)
    print("R2:{}".format(score))

    # MSE评价指标
    mse_test = mean_squared_error(y_test, y_predict)
    print("MSE:{}".format(mse_test))

    # RMSE评价指标
    Rmse_test = mean_absolute_error(y_test, y_predict)
    print("RMSE:{}".format(Rmse_test))


# 模型训练调参
def train_model(model, params_dict, X_train, y_train):
    grid = GridSearchCV(model, params_dict, cv=5, scoring='r2')
    grid.fit(X_train, y_train)

    print('最佳的参数组合为:')
    print(grid.best_params_)
    print('训练集上的最佳R2为:')
    print(grid.best_score_)

    best_model = grid.best_estimator_

    return best_model

# 模型结果绘图
def plot_model(y_predict, y_test):
    plt.figure(figsize=(10, 6))
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
    plt.rcParams['axes.unicode_minus'] = False
    plt.title('模型结果图')
    plt.xlabel='测试集样本序号'
    plt.ylabel='输出LNC值'
    idx_list = np.arange(len(y_predict))
    plt.plot(idx_list, y_predict, label='y_predict', color='g')
    plt.plot(idx_list, y_test, label='y_true', color='r')
    plt.legend(loc='best', fontsize=15)
    plt.show()


if __name__ == '__main__':
    # 读取数据
    filepath='LNC_data.xlsx'
    df = pd.read_excel(filepath)

    # 预处理
    df = Data_preprocessing(df)

    # 数据拆分
    label = 'Y'
    X = df.loc[:, df.columns != label].values
    Y = df.loc[:, df.columns == label].values
    X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=0, test_size=0.20)

    # 模型调参
    model = RandomForestRegressor()
    params_dict = {'n_estimators': range(10, 71, 10),
                   'max_depth': range(3, 10, 2)}
    best_model = train_model(model, params_dict, X_train, y_train)
    # 模型重训练
    best_model.fit(X_train, y_train)
    # 模型预测
    y_predict = best_model.predict(X_test)

    # 模型结果展示
    print('测试集上的模型表现：')
    evaluate_model(y_predict, y_test)
    plot_model(y_predict, y_test)
