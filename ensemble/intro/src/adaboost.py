"""
2クラス分類である、オリジナルのAdaBoost を使う
classification
python src/adaboost.py -i datasets/sonar.all-data
"""
import numpy as np
from sklearn.tree import DecisionTreeClassifier

import support
from weighted import WeighedDecisionTree, WeighedZeroRule, w_gini


class AdaBoost:
    def __init__(self, boost=5, max_depth=5):
        self.boost = boost
        self.max_depth = max_depth
        self.trees = None
        self.alpha = None

    def fit(self, x, y):
        # ブースティングで使用する変数
        self.trees = []  # 各機械学習モデルの配列
        self.alpha = np.zeros((self.boost,))  # 貢献度の配列
        n_clz = y.shape[1]
        if n_clz != 2:
            return  # 基本のAdaBoostは2クラス分類のみ
        y_bin = y.argmax(axis=1) * 2 - 1  # 1と-1の配列にする
        # 学習データに対する重み
        weights = np.ones((len(x),)) / len(x)

        # ブースティング
        for i in range(self.boost):
            # 決定木モデルを作成
            tree = WeighedDecisionTree(
                max_depth=self.max_depth, metric=w_gini, leaf=WeighedZeroRule
            )
            tree.fit(x, y, weights)
            # 一度、学習データに対して実行する
            z = tree.predict(x)
            z_bin = z.argmax(axis=1) * 2 - 1  # 1と-1の配列にする
            # 正解したデータを探す
            filter = z_bin == y_bin  # 正解データの位置がTrueになる配列
            err = weights[filter == False].sum()  # 不正解の位置にある重みの合計
            print("itre #%d -- error=%f" % (i + 1, err))
            # 終了条件
            if i == 0 and err == 0:  # 最初に完全に学習してしまった
                self.trees.append(tree)  # 最初のモデルだけ
                self.alpha = self.alpha[: i + 1]
                break
            if err >= 0.5 or err == 0:  # 正解率が1/2を下回った
                self.alpha = self.alpha[:i]  # 一つ前まで
                break
            # 学習したモデルを追加
            self.trees.append(tree)
            # AdaBoostの計算
            self.alpha[i] = np.log((1.0 - err) / err) / 2.0  # 式9
            weights *= np.exp(-1.0 * self.alpha[i] * y_bin * z_bin)  # 式10
            weights /= weights.sum()  # 重みの正規化

    def predict(self, x):
        # 各モデルの出力の合計
        z = np.zeros((len(x),))
        for i, tree in enumerate(self.trees):
            p = tree.predict(x)
            p_bin = p.argmax(axis=1) * 2 - 1  # 1と-1の配列にする
            z += p_bin * self.alpha[i]  # 貢献度を加味して追加
        # 合計した出力を、その符号で[0,1]と[1,0]の配列にする
        return np.array([z <= 0, z > 0]).astype(int).T

    def __str__(self):
        s = []
        for i, t in enumerate(self.trees):
            s.append("tree: #%d -- weight=%f" % (i + 1, self.alpha[i]))
            s.append(str(t))
        return "\n".join(s)


if __name__ == "__main__":
    import pandas as pd

    ps = support.get_base_args()
    ps.add_argument("--boost", "-b", type=int, default=5, help="Bum of Boosting")
    ps.add_argument("--depth", "-d", type=int, default=5, help="Max Tree Depth")
    args = ps.parse_args()

    df = pd.read_csv(args.input, sep=args.separator, header=args.header, index_col=args.indexcol)
    #  df.shape (208, 61)

    x = df[df.columns[:-1]].values

    # import ipdb as pdb

    # pdb.set_trace()

    if not args.regression:
        y, clz = support.clz_to_prob(df[df.columns[-1]])
        plf = AdaBoost(boost=args.boost, max_depth=args.depth)
        support.report_classifier(plf, x, y, clz, args.crossvalidate)

    else:
        raise Exception("Regression is not supported with an original AdaBoost.")
