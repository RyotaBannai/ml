import numpy as np

from weighted import WeighedDecisionTree, WeighedZeroRule, w_gini


class AdaBoostM1:
    def __init__(self, boost=5, max_depth=5):
        self.boost = boost
        self.max_depth = max_depth
        self.trees = None
        self.beta = None
        self.n_clz = 0  # クラスの個数

    def fit(self, x, y):
        # ブースティングで使用する変数
        self.trees = []  # 各機械学習モデルの配列
        self.beta = np.zeros((self.boost,))
        self.n_clz = y.shape[1]  # 扱うクラス数
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
            # 正解したデータを探す
            filter = z.argmax(axis=1) == y.argmax(axis=1)  # 正解データの位置がTrueになる配列
            err = weights[filter == False].sum()  # 不正解の位置にある重みの合計
            print("itre #%d -- error=%f" % (i + 1, err))
            # 終了条件
            if i == 0 and err == 0:  # 最初に完全に学習してしまった
                self.trees.append(tree)  # 最初のモデルだけ
                self.beta = self.beta[: i + 1]
                break
            if err >= 0.5 or err == 0:  # 正解率が1/2を下回った
                self.beta = self.beta[:i]  # 一つ前まで
                break
            # 学習したモデルを追加
            self.trees.append(tree)
            # AdaBoost.M1の計算
            self.beta[i] = err / (1.0 - err)  # 式2
            weights[filter] *= self.beta[i]  # 式3
            weights /= weights.sum()  # 重みの正規化

    def predict(self, x):
        # 各モデルの出力の合計
        z = np.zeros((len(x), self.n_clz))
        # 各モデルの貢献度を求める
        w = np.log(1.0 / self.beta)
        if w.sum() == 0:  # 完全に学習してしまいエラーが0の時
            w = np.ones((len(self.trees),)) / len(self.trees)
        # 全てのモデルの貢献度付き合算
        for i, tree in enumerate(self.trees):
            p = tree.predict(x)  # p はクラスの確率を表す二次元配列
            c = p.argmax(axis=1)  # c に分類されたクラスの番号
            for j in range(len(x)):
                z[j, c[j]] += w[
                    i
                ]  # 分類されたクラスの位置に貢献度を加算,「合算値」であることに注意
        return z  # クラスの属する可能性を表す配列として返す

    def __str__(self):
        s = []
        w = np.log(1.0 / self.beta)
        if w.sum() == 0:
            w = np.ones((len(self.trees),)) / len(self.trees)
        for i, t in enumerate(self.trees):
            s.append("tree: #%d -- weight=%f" % (i + 1, w[i]))
            s.append(str(t))
        return "\n".join(s)


if __name__ == "__main__":
    import pandas as pd

    ps = support.get_base_args()
    ps.add_argument("--boost", "-b", type=int, default=5, help="Bum of Boosting")
    ps.add_argument("--depth", "-d", type=int, default=5, help="Max Tree Depth")
    args = ps.parse_args()

    df = pd.read_csv(args.input, sep=args.separator, header=args.header, index_col=args.indexcol)
    x = df[df.columns[:-1]].values

    if not args.regression:
        y, clz = support.clz_to_prob(df[df.columns[-1]])
        plf = AdaBoostM1(boost=args.boost, max_depth=args.depth)
        support.report_classifier(plf, x, y, clz, args.crossvalidate)
