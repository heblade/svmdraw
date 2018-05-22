import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.metrics import accuracy_score


def startjob():
    data = pd.read_csv('./data/bipartition.txt', sep=' ', header=None)
    x, y = data[[0, 1]], data[2]

    #分类器
    clf_param = (('linear', 0.1),
                 ('linear', 0.5),
                 ('linear', 1),
                 ('linear', 2),
                 ('rbf', 1, 0.1),
                 ('rbf', 1, 1),
                 ('rbf', 1, 10),
                 ('rbf', 1, 100),
                 ('rbf', 5, 0.1),
                 ('rbf', 5, 1),
                 ('rbf', 5, 10),
                 ('rbf', 5, 100))
    #输出x矩阵中每列最小值
    x1_min, x2_min = np.min(x, axis=0)
    #输出x矩阵中每列最大值
    x1_max, x2_max = np.max(x, axis=0)
    x1, x2 = np.mgrid[x1_min:x1_max:200j, x2_min:x2_max:200j]
    grid_test = np.stack((x1.flat, x2.flat), axis=1)
    cm_light = mpl.colors.ListedColormap(['#77E0A0', '#FFA0A0'])
    cm_dark = mpl.colors.ListedColormap(['g', 'r'])
    mpl.rcParams['font.sans-serif'] = ['SimHei']
    mpl.rcParams['axes.unicode_minus'] = False
    plt.figure(figsize=(14, 10), facecolor='w')
    for i, param in enumerate(clf_param):
        clf = svm.SVC(C=param[1], kernel=param[0])
        if param[0] == 'rbf':
            clf.gamma = param[2]
            title = '高斯核, C=%.1f, $\gamma$ = %.1f' % (param[1], param[2])
        else:
            title = '线性核, C=%.1f ' % param[1]
        clf.fit(x, y)
        y_hat = clf.predict(x)
        print('准确率: ', accuracy_score(y, y_hat))

        #画图
        print(title)
        print('支撑向量的数目: ', clf.n_support_)
        print('支撑向量的系数: ', clf.dual_coef_)
        print('支撑向量: ', clf.support_)
        plt.subplot(3, 4, i+1)
        grid_hat = clf.predict(grid_test)
        grid_hat = grid_hat.reshape(x1.shape)
        plt.pcolormesh(x1, x2, grid_hat, cmap=cm_light, alpha=0.8)
        plt.scatter(x[0], x[1], c=y, edgecolors='k', s=40, cmap=cm_dark)
        plt.scatter(x.loc[clf.support_, 0],
                    x.loc[clf.support_, 1],
                    edgecolors='k',
                    facecolors='none',
                    s=100,
                    marker='o')
        z = clf.decision_function(grid_test)
        print('z = \n', z)
        print('clf.decision_function(x) = ', clf.decision_function(x))
        print('clf.predict(x) = ', clf.predict(x))
        z = z.reshape(x1.shape)
        # plt.contour(x1, x2, z, colors=list('kbrbk'),
        #             linestyles=['--', '--', '-', '--', '--'],
        #             linewidths=[1, 0.5, 1.5, 0.5, 1],
        #             levels=[-1, -0.5, 0, 0.5, 1])
        plt.contour(x1, x2, z, colors=list('brb'),
                    linestyles=['--','-', '--'],
                    linewidths=[1, 1.5, 1],
                    levels=[-1, 0, 1])
        plt.xlim(x1_min, x1_max)
        plt.ylim(x2_min, x2_max)
        plt.title(title, fontsize=14)
    plt.suptitle('SVM不同参数的分类', fontsize=20)
    plt.tight_layout(1.4)
    plt.subplots_adjust(top=0.92)
    plt.show()


if __name__ == '__main__':
    startjob()