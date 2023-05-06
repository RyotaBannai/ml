"""
regression
python3 src/xgradientboost.py -i datasets/airfoil_self_noise.dat -s '\t' -r -c
"""
import random
from copy import deepcopy

import numpy as np

import entropy
import support
from gradientboost import GradientBoost
from linear import Linear
from pruning import PrunedTree, criticalscore, getscore
from zeror import ZeroRule


class ScoringTree(PrunedTree):
    def __init__(
        self, scoring="mse", max_depth=5, metric=entropy.gini, leaf=ZeroRule, critical=1.0, depth=1
    ):
        super().__init__(
            max_depth=max_depth, metric=metric, leaf=leaf, critical=critical, depth=depth
        )
        self.scoring = scoring  # mse or acc

    def get_node(self):
        # 新しくノードを作成する
        return ScoringTree(
            max_depth=self.max_depth,
            metric=self.metric,
            leaf=self.leaf,
            critical=self.critical,
            depth=self.depth + 1,
        )

    # データごとのスコアの配列
    def get_validation(self, y_pred, y_true):
        # 正解データとの差をスコアにする関数
        s = np.array([])
        if self.scoring == "mse":
            s = (y_pred - y_true) ** 2  # 二乗誤差
        elif self.scoring == "acc":
            # 値が小さいほど良いので不一致の数（1-accuracy）
            s = (y_pred.argmax(axis=1) != y_true.argmax(axis=1)).astype(np.float32)
        return s.reshape((-1,))

    # bef_pred: モデル全体のそれまでの出力.
    # それまでの全モデルの全データに対する出力の総和?
    def leaf_validations(self, x, y, bef_pred, scores):
        # 説明変数から分割した左右のインデックスを取得
        feat = x[:, self.feat_index]
        val = self.feat_val
        l, r = self.make_split(feat, val)
        # 左右を実行して結果を作成する
        if self.left is not None and len(l) > 0:
            if isinstance(self.left, ScoringTree):
                # 枝なら再帰
                self.left.leaf_validations(x[l], y[l], bef_pred[l], scores)
            else:
                # 葉ならスコアを作成
                # 求めたz ではどのデータかまでは記録してない?
                z = bef_pred[l] + self.left.predict(x[l])
                s = self.get_validation(z, y[l])
                scores.append(s)
        if self.right is not None and len(r) > 0:
            if isinstance(self.right, ScoringTree):
                # 枝なら再帰
                self.right.leaf_validations(x[r], y[r], bef_pred[r], scores)
            else:
                # 葉ならスコアを作成
                z = bef_pred[r] + self.right.predict(x[r])
                s = self.get_validation(z, y[r])
                scores.append(s)

    # 全ての葉における、分割されたデータのスコアが入った配列がscores となる.
    # 全てのデータのスコアを平均すれば、通常の損失スコア
    # 葉ごとのスコアを求めれば、正則化項に使用するスコア
    def get_validations(self, x, y, bef_pred):
        scores = []
        self.leaf_validations(x, y, bef_pred, scores)
        return scores


class XGradientBoost(GradientBoost):
    def __init__(
        self,
        boost=5,
        ganma=0.001,
        lambda_l1=0.01,
        lambda_l2=0.001,
        optimizer="momentum",
        eta=0.15,  # Shrinkage(縮小), 学習率
        alpha=0.9,
        bag_frac=0.8,
        feat_frac=1.0,
        tree_params={},
    ):
        super().__init__(
            boost=boost, eta=eta, bag_frac=bag_frac, feat_frac=feat_frac, tree_params=tree_params
        )
        self.optimizer = optimizer
        self.ganma = ganma
        self.lambda_l1 = lambda_l1
        self.lambda_l2 = lambda_l2
        self.alpha = alpha

    """
    一つのモデルを作成する.
    初めにfitさせ、枝をプルーニングしながら、葉の数、l1,l2 を罰則項として加えたときに最も良い木の状態を取り出す.
    """

    # x_val, y_val 検証用のデータセット
    # bef_pred_val 検証用データセットに対する直前のモデルの出力
    def fit_one(self, x, y, x_val, y_val, bef_pred_val):
        # 決定木を一回学習させて返す
        min_score = np.inf
        # プルーニングなしの状態で学習させる
        self.tree_params["critical"] = 1.0
        tree = ScoringTree(**self.tree_params)
        tree.fit(x, y)
        fin_tree = deepcopy(tree)
        # 'critical'プルーニング用の枝のスコア
        score = []
        getscore(tree, score)  # 予め全ての枝のスコアを求めておく
        if len(score) > 0:
            # 正則化を行う
            for score_max in sorted(score)[::-1]:  # スコアが大きい方から取り出す
                if score_max <= 0:
                    break
                # 枝を一つずつプルーニングしてゆく
                criticalscore(tree, score_max)
                # 葉を学習させる
                tree.fit_leaf(x, y)
                # 葉における正解データとの差のスコアを列挙する
                scores = tree.get_validations(x_val, y_val, bef_pred_val)
                loss = np.hstack(scores).mean()  # 全てのデータのスコアの平均
                # 罰則項の計算
                s = [t.mean() ** 2 for t in scores]  # 葉毎の誤差の二乗
                s1 = self.ganma * len(s)  # `len`: 葉の数＝決定木の複雑さ
                s2 = self.lambda_l1 * np.mean(s)  # L1項
                s3 = self.lambda_l2 * np.std(s)  # L2項
                # 検証スコア＋罰則項
                score = loss + s1 + s2 + s3
                if score < min_score:
                    min_score = score
                    fin_tree = deepcopy(tree)
        # 最も良かった状態の決定木を返す
        return fin_tree

    def fit(self, x, y, x_val=None, y_val=None):
        # 検証用データセット
        if x_val is None or y_val is None:
            x_val = x
            y_val = y
        # ブースティングで使用する変数
        self.trees = []  # 各機械学習モデルの配列
        self.feats = []  # 各機械学習モデルで使用する次元
        # 初回の学習
        tree = self.fit_one(x, y, x_val, y_val, np.zeros(y_val.shape))
        cur_data = tree.predict(x)
        cur_data_val = tree.predict(x_val)
        # 勾配を作成する
        cur_grad = self.eta * (y - cur_data)
        delta_grad = np.zeros(y.shape)
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
            tree = self.fit_one(train_x, train_y, x_val, y_val, cur_data_val)
            # 一度、学習データに対して実行する
            cur_data += tree.predict(x)
            cur_data_val += tree.predict(x_val)
            # 勾配を更新する
            if self.optimizer == "sgd":
                cur_grad = self.eta * (y - cur_data)
            elif self.optimizer == "momentum":
                bef_grad = cur_grad.copy()
                cur_grad = self.eta * (y - cur_data) + self.alpha * delta_grad
                delta_grad = cur_grad - bef_grad  # 次回のモメンタムで利用
            # 学習したモデルを追加
            self.trees.append(tree)
            # 勾配が無くなったら終了
            if np.all(cur_grad == 0):
                break


if __name__ == "__main__":
    random.seed(1)
    import pandas as pd

    ps = support.get_base_args()
    ps.add_argument("--boost", "-b", type=int, default=5, help="Num of Boost")
    ps.add_argument("--eta", "-l", type=float, default=0.15, help="Learning Ratio")
    ps.add_argument("--alpha", "-p", type=float, default=0.9, help="Alpha of Momentum SGD")
    ps.add_argument("--bagging_fraction", "-a", type=float, default=0.8, help="Bagging Fraction")
    ps.add_argument("--feature_fraction", "-f", type=float, default=1.0, help="Feature Fraction")
    ps.add_argument("--ganma", "-g", type=float, default=0.001, help="Regularization Ganma")
    ps.add_argument("--lambda_l1", "-1", type=float, default=0.01, help="Regularization L1 Lambda")
    ps.add_argument("--lambda_l2", "-2", type=float, default=0.001, help="Regularization L2 Lambda")
    ps.add_argument("--optimizer", "-o", default="momentum", help="Optimizer Function")
    ps.add_argument("--depth", "-d", type=int, default=5, help="Max Tree Depth")
    args = ps.parse_args()

    df = pd.read_csv(args.input, sep=args.separator, header=args.header, index_col=args.indexcol)
    x = df[df.columns[:-1]].values

    if not args.regression:
        y, clz = support.clz_to_prob(df[df.columns[-1]])
        plf = XGradientBoost(
            boost=args.boost,
            eta=args.eta,
            alpha=args.alpha,
            optimizer=args.optimizer,
            bag_frac=args.bagging_fraction,
            feat_frac=args.feature_fraction,
            tree_params={
                "max_depth": args.depth,
                "metric": entropy.gini,
                "leaf": ZeroRule,
                "scoring": "acc",
            },
        )
        support.report_classifier(plf, x, y, clz, args.crossvalidate)
    else:
        y = df[df.columns[-1]].values.reshape((-1, 1))
        plf = XGradientBoost(
            boost=args.boost,
            eta=args.eta,
            alpha=args.alpha,
            optimizer=args.optimizer,
            bag_frac=args.bagging_fraction,
            feat_frac=args.feature_fraction,
            tree_params={
                "max_depth": args.depth,
                "metric": entropy.deviation,
                "leaf": Linear,
                "scoring": "mse",
            },
        )
        support.report_regressor(plf, x, y, args.crossvalidate)
