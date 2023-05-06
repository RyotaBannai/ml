import numpy as np

import support
from dstump import DecisionStump
from pruning import PrunedTree, criticalscore, getscore
from zeror import ZeroRule

"""
src/entropy.py

def gini(y):
    m = y.sum(axis=0)
    size = y.shape[0]
    e = [(p / size) ** 2 for p in m]
    return 1.0 - np.sum(e)
"""


# 重み付きのMetrics関数
# 分類問題だけを考えると、y は [0,0,1] のように該当するクラスのみ１となる.
# ゆえに1 となっているindex の重みを二乗が求めたい重み付きの確率ということになる.
def w_gini(y, weight):
    i = y.argmax(axis=1)
    clz = set(i)
    score = 0.0
    for val in clz:
        p = weight[i == val].sum()
        score += p**2
    return 1.0 - score


def w_infogain(y, weight):
    i = y.argmax(axis=1)
    clz = set(i)
    score = 0.0
    for val in clz:
        p = weight[i == val].sum()
        if p != 0:
            score += p * np.log2(p)
    return -score


"""
>>> data = np.arange(6).reshape((3, 2))
>>> data
array([[0, 1],
       [2, 3],  
       [4, 5]])
>>> np.average(data, axis=1, weights=[1./4, 3./4])
array([0.75, 2.75, 4.75])
"""


# 重み付きの葉となるモデル
class WeighedZeroRule(ZeroRule):
    def fit(self, x, y, weight):
        # 重み付き平均を取る
        self.r = np.average(y, axis=0, weights=weight)
        return self


class WeighedDecisionStump(DecisionStump):
    def __init__(self, metric=w_infogain, leaf=WeighedZeroRule):
        super().__init__(metric=metric, leaf=leaf)
        self.weight = None

    # DecisionStump#split_tree でデータを分割する際の指標として利用
    def make_loss(self, y1, y2, l, r):
        # yをy1とy2で分割したときのMetrics関数の重み付き合計を返す
        if y1.shape[0] == 0 or y2.shape[0] == 0:
            return np.inf
        # Metrics関数に渡す左右のデータの重み
        w1 = self.weight[l] / np.sum(self.weight[l])  # 重みの正規化
        w2 = self.weight[r] / np.sum(self.weight[r])  # 重みの正規化
        total = y1.shape[0] + y2.shape[0]
        m1 = self.metric(y1, w1) * (y1.shape[0] / total)
        m2 = self.metric(y2, w2) * (y2.shape[0] / total)
        return m1 + m2

    def fit(self, x, y, weight):
        # 左右の葉を作成する
        self.weight = weight  # 重みを保存
        self.left = self.leaf()
        self.right = self.leaf()
        # データを左右の葉に振り分ける
        left, right = self.split_tree(x, y)
        # 重みを付けて左右の葉を学習させる
        if len(left) > 0:
            self.left.fit(x[left], y[left], weight[left] / np.sum(weight[left]))  # 重みの正規化
        if len(right) > 0:
            self.right.fit(x[right], y[right], weight[right] / np.sum(weight[right]))  # 重みの正規化
        return self


class WeighedDecisionTree(PrunedTree, WeighedDecisionStump):
    def __init__(self, max_depth=5, metric=w_gini, leaf=WeighedZeroRule, depth=1):
        super().__init__(max_depth=max_depth, metric=metric, leaf=leaf, depth=depth)
        self.weight = None

    def get_node(self):
        # 新しくノードを作成する
        return WeighedDecisionTree(
            max_depth=self.max_depth, metric=self.metric, leaf=self.leaf, depth=self.depth + 1
        )

    def fit(self, x, y, weight):
        self.weight = weight  # 重みを保存
        # 深さ＝１，根のノードの時のみ
        if self.depth == 1 and self.prunfnc is not None:
            # プルーニングに使うデータ
            x_t, y_t = x, y

        # 決定木の学習・・・"critical"プルーニング時は木の分割のみ
        self.left = self.leaf()
        self.right = self.leaf()
        left, right = self.split_tree(x, y)
        if self.depth < self.max_depth:
            if len(left) > 0:
                self.left = self.get_node()
            if len(right) > 0:
                self.right = self.get_node()
        if self.depth < self.max_depth or self.prunfnc != "critical":
            # 重みを付けて左右の枝を学習させる
            if len(left) > 0:
                self.left.fit(x[left], y[left], weight[left] / np.sum(weight[left]))  # 重みの正規化
            if len(right) > 0:
                self.right.fit(x[right], y[right], weight[right] / np.sum(weight[right]))  # 重みの正規化

        # 深さ＝１，根のノードの時のみ
        if self.depth == 1 and self.prunfnc is not None:
            if self.prunfnc == "critical":
                # 学習時のMetrics関数のスコアを取得する
                score = []
                getscore(self, score)
                # スコアから残す枝の最大スコアを計算
                i = int(round(len(score) * self.critical))
                score_max = sorted(score)[i]
                # プルーニングを行う
                criticalscore(self, score_max)
                # 葉を学習させる
                self.fit_leaf(x, y, weight)

        return self

    def fit_leaf(self, x, y, weight):
        # 説明変数から分割した左右のインデックスを取得
        feat = x[:, self.feat_index]
        val = self.feat_val
        l, r = self.make_split(feat, val)
        # 葉のみを学習させる
        if len(l) > 0:
            if isinstance(self.left, PrunedTree):
                self.left.fit_leaf(x[l], y[l], weight[l])
            else:
                self.left.fit(x[l], y[l], weight[l])
        if len(r) > 0:
            if isinstance(self.right, PrunedTree):
                self.right.fit_leaf(x[r], y[r], weight[r])
            else:
                self.right.fit(x[r], y[r], weight[r])
