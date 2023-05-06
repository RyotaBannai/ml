"""
regression
python src/adaboost_r2.py -i  datasets/winequality-red.csv -s ';' -e 0 -r
"""
import numpy as np

import entropy
import support
from linear import Linear
from pruning import PrunedTree


class AdaBoostR2:
    def __init__(self, boost=5, max_depth=5):
        self.boost = boost
        self.max_depth = max_depth
        self.trees = None
        self.beta = None

    def fit(self, x, y):
        # ブースティングで使用する変数
        _x, _y = x, y  # 引数を待避しておく
        self.trees = []  # 各機械学習モデルの配列
        self.beta = np.zeros((self.boost,))
        # 学習データに対する重み
        weights = np.ones((len(x),)) / len(x)
        # ブースティング
        for i in range(self.boost):
            # 決定木モデルを作成
            tree = PrunedTree(max_depth=self.max_depth, metric=entropy.deviation, leaf=Linear)
            # 重み付きの機械学習モデルを代替するため、重みを確率にしてインデックスを取り出す
            all_idx = np.arange(x.shape[0])  # 全てのデータのインデックス
            p_weight = weights / weights.sum()  # 取り出す確率
            idx = np.random.choice(all_idx, size=x.shape[0], replace=True, p=p_weight)
            # インデックスの位置から学習用データを作成する
            x = _x[idx]
            y = _y[idx]
            # モデルを学習する
            tree.fit(x, y)
            # 一度、学習データに対して実行する
            z = tree.predict(x)
            # 差分の絶対値
            l = np.absolute(z - y).reshape((-1,))
            den = np.max(l)
            if den > 0:
                loss = l / den  # 最大の差が1になるようにする
            err = np.sum(weights * loss)  # ランダムな残差だと期待値が1/2になる
            print("itre #%d -- error=%f" % (i + 1, err))
            # 終了条件
            if i == 0 and err == 0:  # 最初に完全に学習してしまった
                self.trees.append(tree)  # 最初のモデルだけ
                self.beta = self.beta[: i + 1]
                break
            if err >= 0.5 or err == 0:
                # 1/2より小さければ、判断しやすいデータとしにくいデータに傾向があるという事
                self.beta = self.beta[:i]
                break
            self.trees.append(tree)
            # AdaBoost.R2の計算
            self.beta[i] = err / (1.0 - err)  # 式12
            weights *= [np.power(self.beta[i], 1.0 - lo) for lo in loss]  # 式13
            weights /= weights.sum()  # 重みの正規化

    """
    使うモデル数m、データ数nすると、推論結果はn*mだから
    res = [[a1,b1,...n1],[a2,b2,...n2],...,[a,bm,...nm]]
    res.T で一つのデータに対する全モデルの推論結果を１次元にする
    res.T = [[a1,a2,...am],[b1,b2,...bm],...,[n1,n2,...nm]]
    ここで１データの推論結果に注目する
    [a1,a2,...am]=[5.33784502, 5.19452227, 5.22152336, 4.85059391, 5.1528582 ] とすると
    これを推論結果の小さいindex を取り出す
    => np.argsort
    => [3, 4, 1, 2, 0] これは順に小さい順にソートして、元のindex を並べている
    累積和をとって、全体の1/2 以上になるindex を取り出す(). ここでは index2
    ([1.40754048, 3.09975078, `4.86478016`, 6.49888757, 7.94866394])
    index2 のモデルを推論を最終的な推論結果として用いる
    => [3, 4, 1, 2, 0][2]
    => 1 であり, [a1,a2,...am][1] はa2 となるため、モデル2 推論結果a2 が最終的な推論結果となる.

    貢献度は、傾向と反対側にあるモデルの方が貢献度が大きくなるため、全てのモデルの貢献度を合算した値の半分の値(1/2)になるまで貢献度を合算していくと、最もそれらしい出力を返すモデルが選択できる.
    なぜ最もそれらしいモデルを選択するのかは、トレンドの間を連続的にするよりも、モデル一つだけ選択する場合、全ての出力はいずれかのトレンドに属するものとなる可能性が高い、ということになるため.
    """

    def predict(self, x):
        # 各モデルの貢献度を求める
        w = np.log(1.0 / self.beta)
        if w.sum() == 0:
            w = np.ones((len(self.trees),)) / len(self.trees)
        # 各モデルの実行結果を予め求めておく
        pred = [tree.predict(x).reshape((-1,)) for tree in self.trees]
        pred = np.array(pred).T  # 対角にするので(データの個数×モデルの数)になる
        # まずそれぞれのモデルの出力を、小さい順に並べて累積和を取る
        idx = np.argsort(pred, axis=1)  # 小さい順番の並び
        cdf = w[idx].cumsum(axis=1)  # 貢献度を並び順に累積してゆく

        cbf_last = cdf[:, -1].reshape((-1, 1))  # 累積和の最後から合計を取得して整形
        # 下界を求める〜プログラム上は全部計算する
        above = cdf >= (1 / 2) * cbf_last  # これはTrueとFalseの二次元配列になる
        # 下界となる場所のインデックスを探す
        median_idx = above.argmax(axis=1)  # Trueが最初に現れる位置
        # そのインデックスにある出力の場所（最初に並べ替えたから）
        median_estimators = idx[np.arange(len(x)), median_idx]
        # その出力の場所にある実行結果の値を求めて返す
        result = pred[np.arange(len(x)), median_estimators]
        return result.reshape((-1, 1))  # 元の次元の形に戻す

    def __str__(self):
        s = []
        w = np.log(1.0 / self.beta)
        if w.sum() == 0:
            w = np.ones((len(self.trees),)) / len(self.trees)
        else:
            w /= w.sum()
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
    args = ps.parse_args()

    df = pd.read_csv(args.input, sep=args.separator, header=args.header, index_col=args.indexcol)
    x = df[df.columns[:-1]].values

    if args.regression:
        y = df[df.columns[-1]].values.reshape((-1, 1))
        plf = AdaBoostR2(boost=args.boost, max_depth=args.depth)
        support.report_regressor(plf, x, y, args.crossvalidate)
