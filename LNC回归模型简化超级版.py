import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn import model_selection,preprocessing
from sklearn.linear_model import Ridge, RidgeCV,Lasso, LassoCV, ElasticNet, ElasticNetCV
from sklearn.metrics import mean_squared_error,mean_absolute_error
from sklearn.decomposition import PCA
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
# 忽略警告

warnings.filterwarnings('ignore')

# 读取LNC数据集
LNC = pd.read_excel(r'LNC_data.xlsx', sep='')
predictors = LNC.columns[0:-1]
pca = PCA(n_components=6)
X_train, X_test, y_train, y_test = model_selection.train_test_split(LNC[predictors], LNC['Y'],test_size=0.2,
                                                                    random_state=1)

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

print("-----------------------岭回归模型-----------------")
# 设置交叉验证的参数，对于每一个Lambda值，都执行 10重 交叉验证
ridge_cv = RidgeCV(alphas=Lambdas, normalize=True, scoring='neg_mean_squared_error', cv=10)
# 模型拟合
ridge_cv.fit(LNC[predictors], LNC['Y'])
# 返回最佳的lambda值
ridge_best_Lambda = ridge_cv.alpha_
print("岭回归模型最优正则化系数为：" + str(ridge_best_Lambda))
# 返回岭回归系数
print('岭回归模型系数：')
print(pd.Series(index=['Intercept'] + X_train.columns.tolist(), data=[ridge_cv.intercept_] + ridge_cv.coef_.tolist()))

# 测试集验证
ridge_predict = ridge_cv.predict(X_test)
RMSE = np.sqrt(mean_squared_error(y_test, ridge_predict))
print("岭回归模型均方根误差为:" + str(RMSE))

print("-----------------------lasso回归模型-----------------")
# LASSO回归模型的交叉验证
lasso_cv = LassoCV(alphas=Lambdas, normalize=True, n_jobs=-1, cv=10, max_iter=10000)
lasso_cv.fit(LNC[predictors], LNC['Y'])
lasso_best_alpha = lasso_cv.alpha_
print("lasso模型最优正则化系数为：" + str(lasso_best_alpha))
# 返回LASSO回归的系数
print("lasso模型回归系数为：")
print(pd.Series(index=['Intercept'] + X_train.columns.tolist(), data=[lasso_cv.intercept_] + lasso_cv.coef_.tolist()))
# 预测
lasso_predict = lasso_cv.predict(X_test)
# 预测效果验证
RMSE = np.sqrt(mean_squared_error(y_test, lasso_predict))
print("lasso模型均方根误差为" + str(RMSE))


print('--------------------------弹性网络模型--------------------------')
Alphas = np.logspace(-5, 2, 200)
L1_ratios = np.logspace(-5, 0, 200)
Elastic_net_cv = ElasticNetCV(alphas=Alphas, l1_ratio=L1_ratios, normalize=True, n_jobs=-1, cv=10, max_iter=10000)
Elastic_net_cv.fit(LNC[predictors], LNC['Y'])
ElasticNet_best_alpha = Elastic_net_cv.alpha_
print("ElasticNet模型正则化系数为：" + str(ElasticNet_best_alpha))
ElasticNet_best_l1_ratio = Elastic_net_cv.l1_ratio_
print("ElasticNet模型混合比例系数为：" + str(ElasticNet_best_l1_ratio))

# 返回ElaticNet模型回归系数
print("ElaticNet模型回归系数为：")
print(pd.Series(index=['Intercept'] + X_train.columns.tolist(),
                data=[Elastic_net_cv.intercept_] + Elastic_net_cv.coef_.tolist()))
# 预测
Elastic_net_predict = Elastic_net_cv.predict(X_test)
# 预测效果验证
RMSE = np.sqrt(mean_squared_error(y_test, Elastic_net_predict))
print("lasso模型均方根误差为" + str(RMSE))
