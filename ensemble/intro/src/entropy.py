from collections import Counter

import numpy as np


def deviation_org(y):
    d = y - y.mean()
    s = d**2
    return np.sqrt(s.mean())


def gini_org(y):
    i = y.argmax(axis=1)
    clz = set(i)
    c = Counter(i)
    size = y.shape[0]
    score = 0.0
    for val in clz:
        score += (c[val] / size) ** 2
    return 1.0 - score


def infgain_org(y):
    i = y.argmax(axis=1)
    clz = set(i)
    c = Counter(i)
    size = y.shape[0]
    score = 0.0
    for val in clz:
        p = c[val] / size
        if p != 0:
            score += p * np.log2(p)
    return -score


def deviation(y):
    return y.std()


"""
>>> np.array([[1., 0., 0.],[1., 0., 0.]]).sum(axis=0)
array([2., 0., 0.])
"""


# ジニ不純度
def gini(y):
    # 次に不純度はノードに含まれるクラス数n に対して、クラスi に含まれるサンプル数をm とすると
    # m/n の確率として表される.
    # y に入ってくる教師データは元から、全てのクラスの確率を足して1 となるから、
    # （つまり、データがクラスi に含まれる確率はすでに求まっているor推論済み）
    # クラスi に含まれるサンプル数など計算せずここでは、そのまま計算している
    m = y.sum(axis=0)
    size = y.shape[0]
    # データ数分足しているから、size で割って平均を求める.
    # 結局 np.sum(e) でsum するから全体の平均をとっている
    e = [(p / size) ** 2 for p in m]

    import ipdb as pdb

    pdb.set_trace()
    return 1.0 - np.sum(e)


def infgain(y):
    m = y.sum(axis=0)
    size = y.shape[0]
    e = [p * np.log2(p / size) / size for p in m if p != 0.0]
    return -np.sum(e)
