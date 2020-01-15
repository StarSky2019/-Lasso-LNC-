import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn import model_selection,preprocessing
from sklearn.linear_model import Ridge, RidgeCV,Lasso, LassoCV, ElasticNet, ElasticNetCV
from sklearn.metrics import mean_squared_error,mean_absolute_error
from sklearn.decomposition import PCA

# 忽略警告

warnings.filterwarnings('ignore')

# 读取LNC数据集
LNC = pd.read_excel(r'LNC_data.xlsx', sep='')
predictors = LNC.columns[0:-1]
pca = PCA(n_components=6)
X_train, X_test, y_train, y_test = model_selection.train_test_split(LNC[predictors], LNC['Y'],test_size=0.2,
                                                                    random_state=1)
X_pca_train=pca.fit_transform(X_train)
X_pca_test=pca.transform(X_test)
X_train=pd.DataFrame(X_pca_train,columns=['a','b','c','d','e','f'])
X_test=pd.DataFrame(X_pca_test,columns=['a','b','c','d','e','f'])

# 图像异常文字数据显示预处理
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# 绘制不同特征之间的相关系数图像
a=LNC.ix[:,:-2].corr()
mask = np.zeros_like(a)
mask[np.tril_indices_from(mask)] = True
plt.figure(figsize=(7, 6))
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
plt.figure(figsize=[8,7])
plt.title('岭回归模型系数与正则化系数关系曲线图')
plt.style.use('ggplot')
plt.plot(Lambdas, ridge_cofficients)
plt.xscale('log')
plt.xlabel('正则化系数Lambda')
plt.ylabel('岭回归模型系数')
plt.legend(['NDVI','NRI','GNDVI','SIPI','PSRI','DVI','RVI','EVI'],loc='best')
plt.show()

# 绘制LASSO模型不同Lambda与回归系数的关系图
plt.figure(figsize=[7,6])
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


print('--------------------------弹性网络模型--------------------------')
Alphas = np.logspace(-5, 2, 200)
L1_ratios = np.logspace(-5, 0, 200)
Elastic_net_cv = ElasticNetCV(alphas=Alphas, l1_ratio=L1_ratios, normalize=True, n_jobs=-1, cv=10, max_iter=10000)
Elastic_net_cv.fit(X_train, y_train)
ElasticNet_best_alpha = Elastic_net_cv.alpha_
print("ElasticNet模型正则化系数为：" + str(ElasticNet_best_alpha))
ElasticNet_best_l1_ratio = Elastic_net_cv.l1_ratio_
print("ElasticNet模型混合比例系数为：" + str(ElasticNet_best_l1_ratio))

# 基于最佳的参数建模
Elastic_net = ElasticNet(alpha=ElasticNet_best_alpha, l1_ratio=ElasticNet_best_l1_ratio, normalize=True, max_iter=10000)
Elastic_net.fit(X_train, y_train)
# 返回ElaticNet模型回归系数
print("ElaticNet模型回归系数为：")
print(pd.Series(index=['Intercept'] + X_train.columns.tolist(),
                data=[Elastic_net.intercept_] + Elastic_net.coef_.tolist()))
# 预测
Elastic_net_predict = Elastic_net.predict(X_test)
# 预测效果验证
RMSE = np.sqrt(mean_squared_error(y_test, Elastic_net_predict))
print("lasso模型均方根误差为" + str(RMSE))

# 设置存储模型的偏回归系数与RMSE参数
Elastic_Net_cofficients_1 = []
Elastic_Net_error_1 = []
Elastic_Net_cofficients_2 = []
Elastic_Net_error_2 = []

# 迭代不同的Lambda值对应的模型
for alpha in Alphas:
    # 数据进行归一化处理
    Elastic_Net = ElasticNet(alpha=alpha, normalize=True)
    Elastic_Net.fit(X_train, y_train)
    Elastic_Net_predict = Elastic_Net.predict(X_test)
    RMSE1 = np.sqrt(mean_squared_error(y_test, Elastic_Net_predict))
    Elastic_Net_error_1.append(RMSE1)
    Elastic_Net_cofficients_1.append(Elastic_Net.coef_)
for l1_ratio in L1_ratios:
    Elastic_Net = ElasticNet(l1_ratio=l1_ratio, normalize=True, precompute=True, max_iter=10000)
    Elastic_Net.fit(X_train, y_train)
    Elastic_Net_predict = Elastic_Net.predict(X_test)
    RMSE2 = np.sqrt(mean_squared_error(y_test, Elastic_Net_predict))
    Elastic_Net_error_2.append(RMSE2)
    Elastic_Net_cofficients_2.append(Elastic_Net.coef_)

# 绘制弹性网络回归模型不同Lambda与回归系数的关系图
plt.figure(figsize=[9, 8])
plt.title('弹性网络模型回归系数与正则化系数关系曲线图')
plt.style.use('ggplot')
plt.plot(Alphas, Elastic_Net_cofficients_1)
plt.xscale('log')
plt.xlabel('正则化系数Lambda')
plt.ylabel('弹性网络回归模型系数')
plt.legend(['NDVI', 'NRI', 'GNDVI', 'SIPI', 'PSRI', 'DVI', 'RVI', 'EVI'], loc='best')
plt.show()

# 绘制弹性网络回归模型不同Lambda与回归系数的关系图
plt.figure(figsize=[9, 8])
plt.title('弹性网络回归模型系数与l1_ratio关系曲线图')
plt.style.use('ggplot')
plt.plot(L1_ratios, Elastic_Net_cofficients_2)
plt.xscale('log')
plt.xlabel('l1_ratio')
plt.ylabel('弹性网络回归模型系数')
plt.legend(['NDVI', 'NRI', 'GNDVI', 'SIPI', 'PSRI', 'DVI', 'RVI', 'EVI'], loc='best')
plt.show()