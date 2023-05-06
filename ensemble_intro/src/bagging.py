import random

import numpy as np

import entropy
import support
from linear import Linear
from pruning import PrunedTree
from zeror import ZeroRule


class Bagging:
    def __init__(self, n_trees=5, ratio=1.0, tree=PrunedTree, tree_params={}):
        self.n_trees = n_trees
        self.ratio = ratio
        self.tree = tree
        self.tree_params = tree_params
        self.trees = []

    def fit(self, x, y):
        # 機械学習モデル用のデータの数
        n_sample = int(round(len(x) * self.ratio))
        for _ in range(self.n_trees):
            # 重複ありランダムサンプルで学習データへのインデックスを生成する
            index = random.choices(np.arange(len(x)), k=n_sample)
            # 新しい機械学習モデルを作成する
            tree = self.tree(**self.tree_params)
            # 機械学習モデルを一つ学習させる
            tree.fit(x[index], y[index])
            # 機械学習モデルを保存
            self.trees.append(tree)
        return self

    def predict(self, x):
        # 全ての機械学習モデルの結果をリストにする
        z = [tree.predict(x) for tree in self.trees]
        # リスト内の結果の平均をとって返す
        return np.mean(z, axis=0)

    def __str__(self):
        return "\n".join(["tree#%d\n%s" % (i, tree) for i, tree in enumerate(self.trees)])


if __name__ == "__main__":
    random.seed(1)
    import pandas as pd

    ps = support.get_base_args()
    ps.add_argument("--trees", "-t", type=int, default=5, help="Num of Trees")
    ps.add_argument("--ratio", "-p", type=float, default=1.0, help="Bagging size")
    ps.add_argument("--depth", "-d", type=int, default=5, help="Max Tree Depth")
    args = ps.parse_args()

    df = pd.read_csv(args.input, sep=args.separator, header=args.header, index_col=args.indexcol)
    x = df[df.columns[:-1]].values

    if not args.regression:
        y, clz = support.clz_to_prob(df[df.columns[-1]])
        plf = Bagging(
            n_trees=args.trees,
            ratio=args.ratio,
            tree_params={"max_depth": args.depth, "metric": entropy.gini, "leaf": ZeroRule},
        )
        support.report_classifier(plf, x, y, clz, args.crossvalidate)
    else:
        y = df[df.columns[-1]].values.reshape((-1, 1))
        plf = Bagging(
            n_trees=args.trees,
            ratio=args.ratio,
            tree_params={"max_depth": args.depth, "metric": entropy.deviation, "leaf": Linear},
        )
        support.report_regressor(plf, x, y, args.crossvalidate)
