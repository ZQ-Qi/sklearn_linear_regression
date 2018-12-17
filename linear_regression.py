import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

import numpy as np
from sklearn.linear_model import LinearRegression
from scipy import stats


def showScatterPlot(x,y):
    # 绘制散点图
    plt.style.use('ggplot')
    ## 解决中文字符显示不全
    font = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=12)
    plt.scatter(x, y)
    plt.xlabel('szzz', fontproperties=font)
    plt.ylabel("（对数）收益率", fontproperties=font)
    plt.title('散点图', fontproperties=font)
    plt.show()

def sklearnLinear(x,y,prd_x_in):
    # x_in = np.array(x).reshape(1, -1)   # 整理x数据
    x_in = np.array(x).reshape(-1, 1)                  # 整理x数据
    # y_in = np.array(y).reshape(1, -1)   # 整理y数据
    y_in = np.array(y).reshape(-1, 1)                   # 整理y数据
    prd_x_in = np.array(prd_x_in).reshape(-1, 1)
    lreg = LinearRegression()           # 创建回归对象
    lreg.fit(x_in, y_in)                # 进行线性回归
    y_prd = lreg.predict(x_in)          # 使用训练好的模型根据x值预测y值
    prd_y_out = lreg.predict(prd_x_in).tolist()
    print("prd_y 类型：{}".format(type(prd_y_out)))
    print('一元线性回归方程为: ' + '\ty' + '=' + str(lreg.intercept_[0]) + ' + ' + str(lreg.coef_[0][0]) + '*x')
    n = len(x_in)
    Regression = sum((y_prd - np.mean(y_in)) ** 2)  # 回归
    Residual = sum((y_in - y_prd) ** 2)  # 残差
    R_square = Regression / (Regression + Residual)  # 相关性系数R^2
    F = (Regression / 1) / (Residual / (n - 2))  # F 分布
    pf = stats.f.sf(F, 1, n - 2)
    message1 = ('相关系数(R^2)： ' + str(R_square[0]) + '；' + '\n' +
                '回归分析(SSR)： ' + str(Regression[0]) + '；' + '\t残差(SSE)： ' + str(Residual[0]) + '；' + '\n' +
                '           F ： ' + str(F[0]) + '；' + '\t' + 'pf ： ' + str(pf[0]))
    ## T
    L_xx = n * np.var(x)
    sigma = np.sqrt(Residual / n)
    t = lreg.coef_ * np.sqrt(L_xx) / sigma
    pt = stats.t.sf(t, n - 2)
    message2 = '           t ： ' + str(t[0][0]) + '；' + '\t' + 'pt ： ' + str(pt[0][0])
    print(message1 + '\n' + message2)

    # intercept coef r_square regression residual f pf t pt
    return [str(lreg.intercept_[0]), str(lreg.coef_[0][0]), str(R_square[0]), str(Regression[0]), str(Residual[0]), str(F[0]), str(pf[0]), str(t[0][0]), str(pt[0][0]), prd_y_out]


if __name__ == '__main__':
    x = [3.4, 1.8, 4.6, 2.3, 3.1, 5.5, 0.7, 3.0, 2.6, 4.3, 2.1, 1.1, 6.1, 4.8, 3.8]
    y = [26.2, 17.8, 31.3, 23.1, 27.5, 36.0, 14.1, 22.3, 19.6, 31.3, 24.0, 17.3, 43.2, 36.4, 26.1]
    print(sklearnLinear(x, y, [1,2,3]))