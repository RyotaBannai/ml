import numpy as np

import entropy
import support
from linear import Linear
from pruning import PrunedTree


class AdaBoostRT:
    def __init__(self, threshold=0.01, boost=5, max_depth=5):
        self.boost = boost
        self.max_depth = max_depth
        self.trees = None
        self.beta = None
        self.threshold = threshold

    def fit(self, x, y):
        # ブースティングで使用する変数
        _x, _y = x, y  # 引数を待避しておく
        self.trees = []
        self.beta = np.zeros((self.boost,))
        # 学習データに対する重み
        weights = np.ones((len(x),)) / len(x)
        # threshold値
        threshold = self.threshold
        # ブースティング
        for i in range(self.boost):
            # 決定木モデルを作成
            tree = PrunedTree(max_depth=self.max_depth, metric=entropy.deviation, leaf=Linear)
            # 重み付きの機械学習モデルを代替するため、重みを確率にしてインデックスを取り出す
            all_idx = np.arange(x.shape[0])  # 全てのデータのインデックス
            p_weight = weights / weights.sum()  # 取り出す確率
            # 元データと同じ数ランダムサンプリングする
            idx = np.random.choice(all_idx, size=x.shape[0], replace=True, p=p_weight)
            # インデックスの位置から学習用データを作成する
            x = _x[idx]
            y = _y[idx]
            # モデルを学習する
            tree.fit(x, y)
            # 一度、学習データに対して実行する
            z = tree.predict(x)
            # 値の大きさに影響されないよう、相対誤差とする
            l = np.absolute(z - y).reshape((-1,)) / y.mean()
            # 正解に相当するデータを探す
            filter = l < threshold  # 正解に相当するデータの位置がTrueになる配列
            err = weights[filter == False].sum()  # 不正解に相当する位置にある重みの合計
            print("itre #%d -- error=%f" % (i + 1, err))
            # 終了条件
            if err < 1e-10:  # 完全に学習してしまった
                self.beta = self.beta[:i]
                break
            # AdaBoost.RTの計算
            self.trees.append(tree)
            self.beta[i] = err / (1.0 - err)  # 式6
            weights[filter] *= self.beta[i] ** 2  # 式7
            weights /= weights.sum()  # 重みの正規化

    def predict(self, x):
        # 各モデルの出力の合計
        z = np.zeros((len(x), 1))
        # 各モデルの貢献度を求める
        w = np.log(1.0 / self.beta)  # 式8
        # 全てのモデルの貢献度付き合算
        for i, tree in enumerate(self.trees):
            p = tree.predict(x)
            z += p * w[i]  # RT では、合算値ではなくかける
        return z / w.sum()

    def __str__(self):
        s = []
        w = np.log(1.0 / self.beta)
        for i, t in enumerate(self.trees):
            s.append("tree: #%d -- weight=%f" % (i + 1, w[i]))
            s.append(str(t))
        return "\n".join(s)


if __name__ == "__main__":
    np.random.seed(1)
    import pandas as pd

    ps = support.get_base_args()
    ps.add_argument("--boost", "-b", type=int, default=5, help="Bum of Boosting")
    ps.add_argument("--depth", "-d", type=int, default=5, help="Max Tree Depth")
    ps.add_argument("--threshold", "-t", type=float, default=0.01, help="Threshold")
    args = ps.parse_args()

    df = pd.read_csv(args.input, sep=args.separator, header=args.header, index_col=args.indexcol)
    x = df[df.columns[:-1]].values

    if args.regression:
        y = df[df.columns[-1]].values.reshape((-1, 1))
        plf = AdaBoostRT(threshold=args.threshold, boost=args.boost, max_depth=args.depth)
        support.report_regressor(plf, x, y, args.crossvalidate)
