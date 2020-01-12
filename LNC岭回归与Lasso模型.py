import pandas as pd
import numpy as np
from sklearn import model_selection
from sklearn.linear_model import Ridge, RidgeCV
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import seaborn as sns
from sklearn.linear_model import Lasso, LassoCV


# 读取LNC数据集
LNC = pd.read_excel(r'LNC_data.xlsx', sep='')
predictors = LNC.columns[0:-1]
X_train, X_test, y_train, y_test = model_selection.train_test_split(LNC[predictors], LNC['Y'],test_size=0.2, random_state=1)

# 图像异常显示预处理
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# 绘制不同特征之间的相关系数图像
a=LNC.ix[:,:-2].corr()
mask = np.zeros_like(a)
mask[np.tril_indices_from(mask)] = True
plt.figure(figsize=(9, 8))
plt.title('不同特征自相关系数图')
sns.despine(left=True)
sns.heatmap(a, annot=True, fmt=".3g",vmax=1, square=True, cmap='viridis',mask=mask.T)
plt.show()

# 生成不同Lambda值
Lambdas = np.logspace(-5, 2, 200)
# 设置存储模型的偏回归系数与RMSE参数
ridge_cofficients = []
ridge_error=[]
lasso_cofficients = []
lasso_error=[]

# 迭代不同的Lambda值对应的模型
for Lambda in Lambdas:
    # 数据进行归一化处理
    ridge = Ridge(alpha=Lambda, normalize=True)
    ridge.fit(X_train, y_train)
    ridge_predict = ridge.predict(X_test)
    RMSE1 = np.sqrt(mean_squared_error(y_test, ridge_predict))
    ridge_error.append(RMSE1)
    ridge_cofficients.append(ridge.coef_)
for Lambda in Lambdas:
    lasso = Lasso(alpha=Lambda, normalize=True, precompute=True, max_iter=10000)
    lasso.fit(X_train, y_train)
    lasso_predict = lasso.predict(X_test)
    RMSE2 = np.sqrt(mean_squared_error(y_test, lasso_predict))
    lasso_error.append(RMSE2)
    lasso_cofficients.append(lasso.coef_)

# 绘制岭回归模型不同Lambda与回归系数的关系图
plt.figure(figsize=[9,10])
plt.title('岭回归模型系数与正则化系数关系曲线图')
plt.style.use('ggplot')
plt.plot(Lambdas, ridge_cofficients)
plt.xscale('log')
plt.xlabel('正则化系数Lambda')
plt.ylabel('岭回归模型系数')
plt.legend(['NDVI','NRI','GNDVI','SIPI','PSRI','DVI','RVI','EVI'],loc='best')
plt.show()

# 绘制LASSO模型不同Lambda与回归系数的关系图
plt.figure(figsize=[9,10])
plt.title('LASSO模型系数与正则化系数关系曲线图')
plt.style.use('ggplot')
plt.plot(Lambdas, lasso_cofficients)
plt.xscale('log')
plt.xlabel('正则化系数Lambda')
plt.ylabel('岭回归模型系数')
plt.legend(['NDVI','NRI','GNDVI','SIPI','PSRI','DVI','RVI','EVI'],loc='best')
plt.show()

# 绘制2个模型Lambda与RMSE的关系图
plt.title('岭回归/Lasso模型系数与RMSE曲线图')
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
plt.style.use('ggplot')
plt.plot(Lambdas, ridge_error)
plt.plot(Lambdas, lasso_error)
# 对x轴作对数变换
plt.xscale('log')
plt.xlabel('Lambda')
plt.ylabel('RMSE')
plt.legend(['岭回归模型','lasso模型'],loc='best')
plt.show()

print("-----------------------岭回归模型-----------------")
# 设置交叉验证的参数，对于每一个Lambda值，都执行 10重 交叉验证
ridge_cv = RidgeCV(alphas=Lambdas, normalize=True, scoring='neg_mean_squared_error', cv=10)
# 模型拟合
ridge_cv.fit(X_train, y_train)
# 计算每一个模型得分
a = ridge_cv.score(X_train, y_train)
# 返回最佳的lambda值
ridge_best_Lambda = ridge_cv.alpha_
print("岭回归模型最优正则化系数为：" + str(ridge_best_Lambda))

# 基于最佳的Lambda值建模
ridge = Ridge(alpha=ridge_best_Lambda, normalize=True)
ridge.fit(X_train, y_train)
# 返回岭回归系数
print('岭回归模型系数：')
print(pd.Series(index=['Intercept'] + X_train.columns.tolist(), data=[ridge.intercept_] + ridge.coef_.tolist()))

# 测试集验证
ridge_predict = ridge.predict(X_test)
RMSE = np.sqrt(mean_squared_error(y_test, ridge_predict))
print("岭回归模型均方根误差为:" + str(RMSE))

print("-----------------------lasso回归模型-----------------")
# LASSO回归模型的交叉验证
lasso_cv = LassoCV(alphas=Lambdas, normalize=True, n_jobs=-1, cv=10, max_iter=10000)
lasso_cv.fit(X_train, y_train)
lasso_best_alpha = lasso_cv.alpha_
print("lasso模型最优正则化系数为：" + str(lasso_best_alpha))

# 基于最佳的lambda值建模
lasso = Lasso(alpha=lasso_best_alpha, normalize=True, max_iter=10000)
lasso.fit(X_train, y_train)
# 返回LASSO回归的系数
print("lasso模型回归系数为：")
print(pd.Series(index=['Intercept'] + X_train.columns.tolist(), data=[lasso.intercept_] + lasso.coef_.tolist()))
# 预测
lasso_predict = lasso.predict(X_test)
# 预测效果验证
RMSE = np.sqrt(mean_squared_error(y_test, lasso_predict))
print("lasso模型均方根误差为" + str(RMSE))
