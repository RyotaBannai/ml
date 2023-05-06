"""
classification
python src/pruning.py -i  datasets/iris.data
regression
python src/pruning.py -i  datasets/airfoil_self_noise.dat  -s '\t' -r -p critical
"""
import numpy as np

import entropy
import support
from dtree import DecisionTree
from linear import Linear
from zeror import ZeroRule

"""
枝刈りする２パターン
1. 右or左の枝を減らした時に、性能劣化しならを削る(もし右だけで性能劣化しないなら、右を本ノードとして返す、など)
2. 右or左だけにデータを割り振れるなら、一方だけのノードを返す本ノードとして返す
"""


def reducederror(node, x, y):
    # ノードが葉でなかったら
    if isinstance(node, PrunedTree):
        # 左右の分割を得る
        feat = x[:, node.feat_index]
        val = node.feat_val
        l, r = node.make_split(feat, val)
        # 左右にデータが振り分けられるか
        if val is np.inf or len(r) == 0:
            return reducederror(node.left, x, y)  # 一つの枝のみの場合、その枝で置き換える
        elif len(l) == 0:
            return reducederror(node.right, x, y)  # 一つの枝のみの場合、その枝で置き換える
        # 左右の枝を更新する
        node.left = reducederror(node.left, x[l], y[l])
        node.right = reducederror(node.right, x[r], y[r])
        # 学習データに対するスコアを計算する
        p1 = node.predict(x)
        p2 = node.left.predict(x)
        p3 = node.right.predict(x)
        # クラス分類かどうか
        if y.shape[1] > 1:
            # 誤分類の個数をスコアにする
            ya = y.argmax(axis=1)
            d1 = np.sum(p1.argmax(axis=1) != ya)
            d2 = np.sum(p2.argmax(axis=1) != ya)
            d3 = np.sum(p3.argmax(axis=1) != ya)
        else:
            # 二乗平均誤差をスコアにする
            d1 = np.mean((p1 - y) ** 2)
            d2 = np.mean((p2 - y) ** 2)
            d3 = np.mean((p3 - y) ** 2)
        if d2 <= d1 or d3 <= d1:  # 左右の枝どちらかだけでスコアが悪化しない
            # スコアの良い方の枝を返す
            if d2 < d3:
                return node.left
            else:
                return node.right
    # 現在のノードを返す
    return node


def getscore(node, score):
    # ノードが葉でなかったら
    if isinstance(node, PrunedTree):
        if node.score >= 0 and node.score is not np.inf:
            score.append(node.score)
        getscore(node.left, score)
        getscore(node.right, score)


def criticalscore(node, score_max):
    # ノードが葉でなかったら
    if isinstance(node, PrunedTree):
        # 左右の枝を更新する
        node.left = criticalscore(node.left, score_max)
        node.right = criticalscore(node.right, score_max)
        # ノードを削除
        if node.score > score_max:
            leftisleaf = not isinstance(node.left, PrunedTree)
            rightisleaf = not isinstance(node.right, PrunedTree)
            # 両方共葉ならば一つの葉にする
            if leftisleaf and rightisleaf:
                return node.left
            # どちらかが枝ならば枝の方を残す
            elif leftisleaf and not rightisleaf:
                return node.right
            elif not leftisleaf and rightisleaf:
                return node.left
            # どちらも枝ならばスコアの良い方を残す
            elif node.left.score < node.right.score:
                return node.left
            else:
                return node.right
    # 現在のノードを返す
    return node


class PrunedTree(DecisionTree):
    def __init__(
        self,
        prunfnc="critical",
        pruntest=False,
        splitratio=0.5,
        critical=0.8,
        max_depth=5,
        metric=entropy.gini,
        leaf=ZeroRule,
        depth=1,
    ):
        super().__init__(max_depth=max_depth, metric=metric, leaf=leaf, depth=depth)
        self.prunfnc = prunfnc  # プルーニング用関数
        self.pruntest = pruntest  # プルーニング用にテスト用データを取り分けるか
        self.splitratio = splitratio  # プルーニング用テストデータの割合
        self.critical = critical  # "critical"プルーニング用の閾値

    def get_node(self):
        # 新しくノードを作成する
        return PrunedTree(
            prunfnc=self.prunfnc,
            max_depth=self.max_depth,
            metric=self.metric,
            leaf=self.leaf,
            depth=self.depth + 1,
        )

    def fit_leaf(self, x, y):
        # 説明変数から分割した左右のインデックスを取得
        feat = x[:, self.feat_index]
        val = self.feat_val
        l, r = self.make_split(feat, val)
        # 葉のみを学習させる
        if len(l) > 0:
            if isinstance(self.left, PrunedTree):
                self.left.fit_leaf(x[l], y[l])
            else:
                self.left.fit(x[l], y[l])
        if len(r) > 0:
            if isinstance(self.right, PrunedTree):
                self.right.fit_leaf(x[r], y[r])
            else:
                self.right.fit(x[r], y[r])

    def fit(self, x, y):
        # 深さ＝１，根のノードの時のみ
        if self.depth == 1 and self.prunfnc is not None:
            # プルーニングに使うデータ
            x_t, y_t = x, y
            # プルーニング用にテスト用データを取り分けるならば
            if self.pruntest:
                # 学習データとテスト用データを別にする
                n_test = int(round(len(x) * self.splitratio))
                n_idx = np.random.permutation(len(x))
                tmpx = x[n_idx[n_test:]]
                tmpy = y[n_idx[n_test:]]
                x_t = x[n_idx[:n_test]]
                y_t = y[n_idx[:n_test]]
                x = tmpx
                y = tmpy

        # 決定木の学習・・・"critical"プルーニング時は木の分割のみ
        self.left = self.leaf()
        self.right = self.leaf()
        left, right = self.split_tree(x, y)
        if self.depth < self.max_depth:
            if len(left) > 0:
                self.left = self.get_node()
            if len(right) > 0:
                self.right = self.get_node()
        # critical なら葉のfit はやらない
        if self.depth < self.max_depth or self.prunfnc != "critical":
            if len(left) > 0:
                self.left.fit(x[left], y[left])
            if len(right) > 0:
                self.right.fit(x[right], y[right])

        # 深さ＝１，根のノードの時のみ
        if self.depth == 1 and self.prunfnc is not None:
            if self.prunfnc == "reduce":
                # プルーニングを行う
                # 全部fit した後
                reducederror(self, x_t, y_t)
            elif self.prunfnc == "critical":
                # 葉をfit する前
                # 学習時のMetrics関数のスコアを取得する
                score = []
                getscore(self, score)
                if len(score) > 0:
                    # スコアから残す枝の最大スコアを計算
                    i = int(round(len(score) * self.critical))
                    score_max = sorted(score)[min(i, len(score) - 1)]
                    # プルーニングを行う
                    criticalscore(self, score_max)
                # 葉を学習させる
                self.fit_leaf(x, y)

        return self


if __name__ == "__main__":
    import pandas as pd

    np.random.seed(1)
    ps = support.get_base_args()
    ps.add_argument("--depth", "-d", type=int, default=5, help="Max Tree Depth")
    ps.add_argument("--test", "-t", action="store_true", help="Test split for pruning")
    ps.add_argument("--pruning", "-p", default="critical", help="Pruning Algorithm")
    ps.add_argument("--ratio", "-a", type=float, default=0.5, help="Test size for pruning")
    ps.add_argument("--critical", "-l", type=float, default=0.8, help="Value for Critical Pruning")
    args = ps.parse_args()

    df = pd.read_csv(args.input, sep=args.separator, header=args.header, index_col=args.indexcol)
    x = df[df.columns[:-1]].values

    if not args.regression:
        y, clz = support.clz_to_prob(df[df.columns[-1]])
        mt = entropy.gini
        lf = ZeroRule
        plf = PrunedTree(
            prunfnc=args.pruning,
            pruntest=args.test,
            splitratio=args.ratio,
            critical=args.critical,
            metric=mt,
            leaf=lf,
            max_depth=args.depth,
        )
        support.report_classifier(plf, x, y, clz, args.crossvalidate)
    else:
        y = df[df.columns[-1]].values.reshape((-1, 1))
        mt = entropy.deviation
        lf = Linear
        plf = PrunedTree(
            prunfnc=args.pruning,
            pruntest=args.test,
            splitratio=args.ratio,
            critical=args.critical,
            metric=mt,
            leaf=lf,
            max_depth=args.depth,
        )
        support.report_regressor(plf, x, y, args.crossvalidate)
