import random

import numpy as np

import entropy
import support
from adaboost_m1 import AdaBoostM1
from adaboost_r2 import AdaBoostR2
from bagging import Bagging
from dstump import DecisionStump
from gradientboost import GradientBoost
from linear import Linear
from modelselect import BICSelect, CVSelect
from randomforest import RandomTree


class StackingMean(CVSelect):
    def __init__(self, isregression, max_depth=5, n_trees=5):
        super().__init__(isregression=isregression, max_depth=max_depth, n_trees=n_trees)

    def fit(self, x, y):
        # 全てのモデルを学習させる
        for j in range(len(self.clf)):
            self.clf[j].fit(x, y)
        return self

    def predict(self, x):
        # 全てのモデルを実行する
        result = []
        for j in range(len(self.clf)):
            result.append(self.clf[j].predict(x))
        # 全てのモデルの平均を返す
        return np.array(result).mean(axis=0)

    def __str__(self):
        return "\n".join([str(c) for c in self.clf])


class NFoldMean:
    def __init__(self, isregression, model=StackingMean, max_depth=5, n_trees=5):
        self.n_fold = 5
        self.clf = [
            model(isregression=isregression, max_depth=max_depth, n_trees=n_trees)
            for i in range(self.n_fold)
        ]

    def fit(self, x, y):
        # 交差検証による選択
        perm_indexs = np.random.permutation(x.shape[0])
        indexs = np.array_split(perm_indexs, self.n_fold)
        # 交差検証を行う
        for i in range(self.n_fold):
            # 学習用データを分割する
            ti = list(range(self.n_fold))
            ti.remove(i)
            train = np.hstack([indexs[t] for t in ti])
            """
            サンプコードでは、clf でさらにループしているけど、そうした場合fit は別のデータセットで繰り返し学習できないから、n_fold でデータセットを変えて学習させる時(ここではi分ループ)には最後のデータセットで全てのモデルが学習させれてしまう。つまり、predict する時は最後のデータセットで学習されるから、複数のモデルを作っても同じ推論結果に収束してしまうため、アンサンブルとしての強みがない。
            おそらくここのでのコードは、それぞれのデータセットで別々のモデルを学習させたいからi分のループでclf を取り出してfit するのが良い.
            """
            # test = indexs[i]
            # 全てのモデルを検証する
            # for j in range(len(self.clf)):
            # 分割したデータで学習
            self.clf[i].fit(x[train], y[train])
        return self

    def predict(self, x):
        # 全てのモデルを実行する
        result = []
        for j in range(len(self.clf)):
            result.append(self.clf[j].predict(x))
        # 全てのモデルの平均を返す
        return np.array(result).mean(axis=0)

    def __str__(self):
        return "\n".join([str(c) for c in self.clf])


class SmoothedBICMean(BICSelect):
    def __init__(self, isregression, max_depth=5, n_trees=5):
        super().__init__(isregression=isregression, max_depth=max_depth, n_trees=n_trees)
        self.bic_scores = None

    def fit(self, x, y):
        # 交差検証を行う
        self.bic_scores = np.zeros((len(self.clf),))
        predicts = self.cv(x, y)
        n_fold = len(predicts) // len(self.clf)
        # 交差検証の結果を取得
        for i in range(n_fold):
            for j in range(len(self.clf)):
                # 評価スコアを尤度関数の代わりに使用する
                p = predicts.pop(0)
                score = self.metric(p[0], p[1])
                # 独立変数の数として葉の総数を使用する
                n_leafs = self.get_totalleaf()
                # 罰則項を加えたスコアで計算
                self.bic_scores[j] += x.shape[0] * np.log(score + 1e-9) + n_leafs * np.log(
                    x.shape[0]
                )
        # 全てのモデルを学習させる
        for j in range(len(self.clf)):
            self.clf[j].fit(x, y)
        return self

    def predict(self, x):
        # スコアは小さい方が良い値なので、最大値から引く
        scores = np.max(self.bic_scores) - self.bic_scores
        # 合算が1になるようにする
        weights = scores / np.sum(scores)
        # 全てのモデルを実行する
        result = []
        for j in range(len(self.clf)):
            result.append(self.clf[j].predict(x))
        # 全てのモデルの重み付き平均を返す
        return np.average(np.array(result), axis=0, weights=weights)


if __name__ == "__main__":
    random.seed(1)
    np.random.seed(1)
    import pandas as pd

    ps = support.get_base_args()
    ps.add_argument("--trees", "-t", type=int, default=5, help="Num of Tree")
    ps.add_argument("--depth", "-d", type=int, default=5, help="Max Tree Depth")
    ps.add_argument(
        "--method", "-m", default="stacking", help="Use Method (stacking / nfold / bic)"
    )
    args = ps.parse_args()

    df = pd.read_csv(args.input, sep=args.separator, header=args.header, index_col=args.indexcol)
    x = df[df.columns[:-1]].values

    if args.method == "stacking":
        plf = StackingMean(isregression=args.regression)
    elif args.method == "nfold":
        plf = NFoldMean(isregression=args.regression)
    elif args.method == "bic":
        plf = SmoothedBICMean(isregression=args.regression)

    if not args.regression:
        y, clz = support.clz_to_prob(df[df.columns[-1]])
        support.report_classifier(plf, x, y, clz, args.crossvalidate)
    else:
        y = df[df.columns[-1]].values.reshape((-1, 1))
        support.report_regressor(plf, x, y, args.crossvalidate)
