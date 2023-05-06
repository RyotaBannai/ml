"""
勾配ブースティング、グラディエントブースティング、GBDT
新しいモデルが学習するのは、前回の教師データと推論結果の誤差.
誤差を是正する（残差を求める）ためのモデルが積み上がっていくから、
それら全てのモデルの推論結果の総和は教師データに近い出力となる.

regression
python src/gradientboost.py -i  datasets/winequality-red.csv -s ';' -e 0 -r
"""
import random

import numpy as np

import entropy
import support
from linear import Linear
from pruning import PrunedTree
from zeror import ZeroRule


class GradientBoost:
    def __init__(self, boost=5, eta=0.3, bag_frac=0.8, feat_frac=1.0, tree_params={}):
        self.boost = boost
        self.eta = eta
        self.bag_frac = bag_frac  # バギング、データ大きさ
        self.feat_frac = feat_frac  # バギング、次元の取り出す大きさ
        self.tree_params = tree_params
        self.trees = None
        self.feats = None

    def fit(self, x, y):
        # ブースティングで使用する変数
        self.trees = []  # 各機械学習モデルの配列
        self.feats = []  # 各機械学習モデルで使用する次元
        # 初回の学習
        tree = PrunedTree(**self.tree_params)
        tree.fit(x, y)
        cur_data = tree.predict(x)
        # 勾配を作成する
        cur_grad = self.eta * (y - cur_data)
        # 学習したモデルを追加
        self.trees.append(tree)
        self.feats.append(np.arange(x.shape[1]))
        # ブースティング
        for i in range(self.boost - 1):
            # バギング
            train_x = x
            train_y = cur_grad
            if self.feat_frac < 1.0:
                # 説明変数内の次元から、ランダムに使用する次元を選択する
                features = int(round(x.shape[1] * self.feat_frac))
                index = random.sample(range(x.shape[1]), features)
                train_x = x[:, index]
                self.feats.append(index)
            else:
                self.feats.append(np.arange(x.shape[1]))
            if self.bag_frac < 1.0:
                # 説明変数から、ランダムに選択する
                baggings = int(round(x.shape[0] * self.bag_frac))
                index = random.sample(range(x.shape[0]), baggings)
                train_x = train_x[index]
                train_y = train_y[index]
            # 勾配を目的変数にして学習する
            tree = PrunedTree(**self.tree_params)
            tree.fit(train_x, train_y)

            # 一度、学習データに対して実行する
            cur_data += tree.predict(x)
            # 勾配を更新する
            cur_grad = self.eta * (y - cur_data)
            # 学習したモデルを追加
            self.trees.append(tree)
            # 勾配が無くなったら終了
            if np.all(cur_grad == 0):
                break

    def predict(self, x):
        # 各モデルの出力の合計
        z = [tree.predict(x[:, f]) for tree, f in zip(self.trees, self.feats)]
        return np.sum(z, axis=0)

    def __str__(self):
        s = []
        for i, t in enumerate(self.trees):
            s.append("tree: #%d" % (i + 1,))
            s.append(str(t))
        return "\n".join(s)


if __name__ == "__main__":
    random.seed(1)
    import pandas as pd

    ps = support.get_base_args()
    ps.add_argument("--boost", "-b", type=int, default=5, help="Num of Boost")
    ps.add_argument("--eta", "-l", type=float, default=0.3, help="Learning Ratio")
    ps.add_argument("--bagging_fraction", "-a", type=float, default=0.8, help="Bagging Fraction")
    ps.add_argument("--feature_fraction", "-f", type=float, default=1.0, help="Feature Fraction")
    ps.add_argument("--depth", "-d", type=int, default=5, help="Max Tree Depth")
    args = ps.parse_args()

    df = pd.read_csv(args.input, sep=args.separator, header=args.header, index_col=args.indexcol)
    x = df[df.columns[:-1]].values

    if not args.regression:
        y, clz = support.clz_to_prob(df[df.columns[-1]])
        plf = GradientBoost(
            boost=args.boost,
            eta=args.eta,
            bag_frac=args.bagging_fraction,
            feat_frac=args.feature_fraction,
            tree_params={"max_depth": args.depth, "metric": entropy.gini, "leaf": ZeroRule},
        )
        support.report_classifier(plf, x, y, clz, args.crossvalidate)
    else:
        y = df[df.columns[-1]].values.reshape((-1, 1))
        plf = GradientBoost(
            boost=args.boost,
            eta=args.eta,
            bag_frac=args.bagging_fraction,
            feat_frac=args.feature_fraction,
            tree_params={"max_depth": args.depth, "metric": entropy.deviation, "leaf": Linear},
        )
        support.report_regressor(plf, x, y, args.crossvalidate)
