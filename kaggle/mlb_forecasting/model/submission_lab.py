"""
Code Competition 用のファイル
実験用.
うまく行ったことを確認したら、この_lab の内容をもとに
正式にsubmission.py で提出コードを構築
"""

# %%

import datetime as dt
import gc
import pickle
import warnings
from collections import defaultdict
from typing import Any, Callable, Dict, Iterable, List, Sequence, Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow
import pyarrow.csv
from dateutil.relativedelta import relativedelta
from sklearn.metrics import mean_absolute_error

warnings.filterwarnings("ignore")
pd.options.display.float_format = "{:10.4f}".format


# %%


def unpack(json_str):
    return np.nan if pd.isna(json_str) else pd.read_json(json_str)


def extract_data(input_df, col, show=False):
    ret_df = pd.DataFrame()
    for i in np.arange(len(input_df)):
        if show:
            print("\r{}/{}".format(i, len(input_df)), end="")
        try:
            ret_df = pd.concat([ret_df, unpack(input_df[col].iloc[i])])
        except:
            if show:
                print("Json Parse Error")
    if show:
        print("")
        print(ret_df.shape)
        print(ret_df.head())
    return ret_df


ROOT = "../data/"
# ROOT = '/kaggle/input/mlb-player-digital-engagement-forecasting/'
train_path = ROOT + "train_updated.csv"
# https://issues.apache.org/jira/browse/ARROW-13314
# https://arrow.apache.org/docs/python/generated/pyarrow.Table.html#pyarrow.Table.slice
csv_ro = pyarrow.csv.ReadOptions(block_size=2**25)
table = pyarrow.csv.read_csv(train_path, read_options=csv_ro)
train_df = table.to_pandas()

# json_ro = pyarrow.csv.ReadOptions(block_size=2**25)
# %%

# 処理時間の高速化
# 季節性を考慮して最低でも1年間+検証データで13ヶ月分を確保.
train = train_df[train_df.date >= 20200401].reset_index(drop=True)
print(train_df[train_df.date >= 20200401].date.nunique() / 365)


# %%
"""
json をパース
"""
df_engagement = extract_data(train, col="nextDayPlayerEngagement", show=True)

# %%
"""
前処理
"""
df_engagement["date_playerId"] = (
    df_engagement["engagementMetricsDate"].str.replace("-", "")
    + "_"
    + df_engagement["playerId"].astype(str)
)

df_engagement.head()
# 推論実施日に関する特徴量を作成
# 推論実施日のカラム作成（推論実施日=推論対象日の前日）✨
# ここで入ってるdate は推論対象日なので、推論実施日にするために-1 日する
df_engagement["date"] = pd.to_datetime(
    df_engagement.engagementMetricsDate, format="%Y-%m-%d"
) + dt.timedelta(days=-1)
df_engagement["dayofweek"] = df_engagement["date"].dt.dayofweek
df_engagement["yearmonth"] = df_engagement["date"].astype(str).apply(lambda s: s[:7])


"""
Players の情報を読み込んで、予測対象か否かのカラム01 へ変換.
"""

df_players = pd.read_csv(ROOT + "players.csv")
df_players = df_players[(df_players.playerId != False)]

# <NA> のままでも np.where 判定できると思う.
df_players.loc[
    df_players.playerForTestSetAndFuturePreds.isna(), "playerForTestSetAndFuturePreds"
] = False
df_players.playerForTestSetAndFuturePreds = (
    df_players.playerForTestSetAndFuturePreds.astype(bool)
)
df_players.playerForTestSetAndFuturePreds = np.where(
    df_players.playerForTestSetAndFuturePreds, 1, 0
)


# プレイヤー情報が一意だから、playerId でmerge しても大丈夫そう？
assert all(
    (df_players.groupby("playerId").agg(func="count").playerName == 1).values.tolist()
)
# 念の為、drop
df_players = df_players.drop_duplicates(["playerId"]).reset_index()
df_train = pd.merge(df_engagement, df_players, on="playerId", how="left")
use_cols = [
    "playerId",
    "dayofweek",
    "birthCity",
    "birthStateProvince",
    "birthCountry",
    "heightInches",
    "weight",
    "primaryPositionCode",
    "primaryPositionName",
    "playerForTestSetAndFuturePreds",
]
target_cols = ["target1", "target2", "target3", "target4"]
id_cols = [
    "engagementMetricsDate",
    "playerId",
    "date_playerId",
    "date",
    "yearmonth",
    "playerForTestSetAndFuturePreds",
]
x_train = df_train[use_cols]
y_train = df_train[target_cols]
id_train = df_train[id_cols]

# LightGBM ではobject 型は使えないため、category 型へconvert しておく.
cat_cols = [
    "playerId",
    "dayofweek",
    "birthCity",
    "birthStateProvince",
    "birthCountry",
    "primaryPositionCode",
    "primaryPositionName",
]
for col in cat_cols:
    x_train[col] = x_train[col].astype("category")

# %%

"""
時系列用のバリデーションを作成
１年間の学習データと、その翌月をテストデータとした
データセットを１ヶ月ごとにずらしたものを３つ用意
"""
list_cv_month = []
ymf = "%Y-%m"
for start in ["2020-05", "2020-06", "2020-07"]:
    # 一年後を計算
    # 年月単位では、relativedelta を使う
    # https://dateutil.readthedocs.io/en/latest/
    end_date = dt.datetime.strptime(start, ymf) + relativedelta(years=1)
    end = end_date.strftime(ymf)
    # end はexclusive で含まれなため注意
    months = pd.date_range(start=start, end=end, freq="M").format(
        formatter=lambda x: x.strftime(ymf)
    )
    # exclusive で含まれない最後の月をvalid として使う
    valid_date = end_date
    valid = valid_date.strftime(ymf)
    list_cv_month.append([months, [valid]])

# ここで、list_cv_month を一度確認する

# valid は予測対象の選手のみを扱う
# 学習データは多い方がいいので含める
# 学習データから、index で取り出せるようにここでは、.index に対してslicing
cv = []
for month_tr, month_va in list_cv_month:
    cv.append(
        [
            id_train.index[id_train.yearmonth.isin(month_tr)],
            id_train.index[
                (id_train.yearmonth.isin(month_va))
                & (id_train.playerForTestSetAndFuturePreds == 1)
            ],
        ]
    )

# %%


def lightgbm_hy_params(trial=None, update_params=None, is_optimize=False) -> dict:
    print("---")
    print("lightgbm params")
    print("{}, {}, {}".format(trial, update_params, is_optimize))
    print("---")

    # ハイパーパラメータ
    params_base = {
        "boosting_type": "gbdt",
        "objective": "regression_l1",
        "metric": "mean_absolute_error",
        "learning_rate": 0.05,
        # "num_iterations": 100000,
        "num_leaves": 32,
        "subsample": 0.7,
        "subsample_freq": 1,
        "feature_fraction": 0.8,
        "min_data_in_leaf": 50,
        "min_sum_hessian_in_leaf": 50,
        "n_estimators": 1000,
        "random_state": 123,
        "importance_type": "gain",
    }

    if update_params is not None:
        params_base.update(update_params)

    if is_optimize and trial is not None:
        # 探索するハイパーパラメータ
        params_tuning = {
            "num_leaves": trial.suggest_int("num_leavels", 8, 256),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 5, 200),
            "min_sum_hessian_in_leaf": trial.suggest_float(
                "min_sum_hessian_in_leaf", 1e-5, 1e-2, log=True
            ),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
            "lambda_l1": trial.suggest_float("lambda_l1", 1e-2, 1e2, log=True),
            "lambda_l2": trial.suggest_float("lambda_l2", 1e-2, 1e2, log=True),
        }
        params_tuning.update(params_base)

    return params_base


targets = [f"target{n}" for n in list(np.arange(4) + 1)]

trained_models: Dict[int, List[Any]] = defaultdict(list)


def train_cv(
    input_x, input_y, input_id, cv: List[List[pd.Index]], mode="load"
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """train with cross validation"""
    metrics = []
    df_valid_pred = pd.DataFrame()
    df_imp = pd.DataFrame()

    for k in np.arange(len(cv)):
        for target in targets:
            """
            target の種類ごとにモデルを作成・予測
            """
            print(f"{k}-fold")
            idx_tr, idx_va = cv[k][0], cv[k][1]
            x_tr, y_tr, _ = (
                input_x.loc[idx_tr, :],
                input_y.loc[idx_tr, target],
                input_id.loc[idx_tr, :],
            )
            x_va, y_va, id_va = (
                input_x.loc[idx_va, :],
                input_y.loc[idx_va, target],
                input_id.loc[idx_va, :],
            )

            fname_lgb = f"./kfold_model_pickle/model_lgb_fold{k}_{target}.pickle"
            if mode == "load":
                # fit済みのモデルを読み込む
                # すでにdump されていない場合、file not found でexception
                # を出してしまうが、今のところ補足したりしてない
                print("model load.")
                with open(fname_lgb, "rb") as f:
                    model = pickle.load(f)
            else:
                verbose_eval = 0
                # https://qiita.com/c60evaporator/items/2b7a2820d575e212bcf4
                callbacks = [
                    # early_stopping用コールバック関数
                    lgb.early_stopping(stopping_rounds=100, verbose=True),
                    # コマンドライン出力用コールバック関数
                    lgb.log_evaluation(verbose_eval),
                ]
                params = lightgbm_hy_params()
                model = lgb.LGBMRegressor(**params)
                model.fit(
                    x_tr,
                    y_tr,
                    eval_set=[(x_tr, y_tr), (x_va, y_va)],
                    callbacks=callbacks,
                )

                trained_models[k].append(model)

            y_va_pred = model.predict(x_va)
            pred_df = pd.DataFrame(
                {
                    "target": target,
                    "nfold": k,
                    "true": y_va,
                    "pred": y_va_pred,
                }
            )
            tmp_valid_pred = pd.concat(
                [id_va, pred_df],
                axis=1,
            )
            df_valid_pred = pd.concat(
                [df_valid_pred, tmp_valid_pred], axis=0, ignore_index=True
            )

            metric_va = mean_absolute_error(y_va, y_va_pred)
            metrics.append([k, target, metric_va])

            print(
                "[mae] kfold: {}, target: {}, va_prob: {:.2f}".format(
                    k, target, metric_va
                )
            )

            _df_imp = pd.DataFrame(
                {
                    "nfold": k,
                    "target": target,
                    "col": x_train.columns,
                    "imp": model.feature_importances_,
                }
            )
            df_imp = pd.concat([df_imp, _df_imp], axis=0, ignore_index=True)

    print("-" * 20, "result", "-" * 20)

    df_metrics = pd.DataFrame(metrics, columns=["nfold", "target", "mae"])
    print("MCMAE: {:.4f}".format(df_metrics.mae.mean()))

    # valid 推論値
    """
    ここで各行の "target{n}_fold{k}_{pred|true}" が１つだけ値があり、他がNaN になっているが、
    これは正しい.
    cv はそれぞれ異なる、月３ヶ月分をvalid として用いているから、
    engagementMetricsDate をindex として、該当の日付がある時点で、それは１つのバリデーション（cv のうちの１つ）の結果しか持っていないことを意味するから、それ以外の fold はNaN
    target は同じfold 内で4つ存在するから、これは同じ位置にある.
    """
    df_valid_pred_all = pd.pivot_table(
        df_valid_pred,
        index=[
            "engagementMetricsDate",
            "playerId",
            "date_playerId",
            "date",
            "yearmonth",
            "playerForTestSetAndFuturePreds",
        ],
        columns=["target", "nfold"],
        values=["true", "pred"],
        aggfunc=np.sum,
    )

    df_valid_pred_all.columns = [
        "{}_fold{}_{}".format(j, k, i) for i, j, k in df_valid_pred_all.columns
    ]
    df_valid_pred_all = df_valid_pred_all.reset_index(drop=False)

    return df_valid_pred_all, df_metrics, df_imp


# %%
"""
train
"""
df_valid_pred, df_metrics, df_imp = train_cv(
    input_x=x_train, input_y=y_train, input_id=id_train, cv=cv, mode=None
)


# %%
"""
入力データを整形し、モデルの入力値の形式に変換
"""


def make_dataset_for_predict(input_test: pd.DataFrame, input_prediction: pd.DataFrame):
    test = input_test.copy()
    prediction = input_prediction.copy()

    prediction.reset_index(drop=False, inplace=True)
    prediction["date"] = pd.to_datetime(prediction.date, format="%Y%m%d")
    prediction["dayofweek"] = prediction["date"].dt.dayofweek
    prediction["yearmonth"] = prediction.date.astype(str).apply(lambda s: s[:7])
    prediction["engagementMetricsDate"] = prediction.date_playerId.astype(str).apply(
        lambda s: s[:8]
    )
    prediction["engagementMetricsDate"] = pd.to_datetime(
        prediction.engagementMetricsDate, format="%Y%m%d"
    )
    prediction["playerId"] = prediction.date_playerId.apply(lambda x: int(x[9:]))

    df_players = pd.read_csv(ROOT + "players.csv")
    df_players = df_players[(df_players.playerId != False)]

    # <NA> のままでも np.where 判定できると思う.
    df_players.loc[
        df_players.playerForTestSetAndFuturePreds.isna(),
        "playerForTestSetAndFuturePreds",
    ] = False
    df_players.playerForTestSetAndFuturePreds = (
        df_players.playerForTestSetAndFuturePreds.astype(bool)
    )
    df_players.playerForTestSetAndFuturePreds = np.where(
        df_players.playerForTestSetAndFuturePreds, 1, 0
    )

    df_train = pd.merge(prediction, df_players, on="playerId", how="left")
    use_cols = [
        "playerId",
        "dayofweek",
        "birthCity",
        "birthStateProvince",
        "birthCountry",
        "heightInches",
        "weight",
        "primaryPositionCode",
        "primaryPositionName",
        "playerForTestSetAndFuturePreds",
    ]
    target_cols = ["target1", "target2", "target3", "target4"]
    id_cols = [
        "engagementMetricsDate",
        "playerId",
        "date_playerId",
        "date",
        "yearmonth",
        "playerForTestSetAndFuturePreds",
    ]
    x_test = df_train[use_cols]
    y_test = df_train[target_cols]
    id_test = df_train[id_cols]

    # LightGBM ではobject 型は使えないため、category 型へconvert しておく.
    cat_cols = [
        "playerId",
        "dayofweek",
        "birthCity",
        "birthStateProvince",
        "birthCountry",
        "primaryPositionCode",
        "primaryPositionName",
    ]
    for col in cat_cols:
        x_test[col] = x_test[col].astype("category")

    return x_test, y_test, id_test


# %%
"""
Code Competition の推論時に受け取るデータを擬似的に作成
これは、Notebook 上でライブラリ MLB を使って取得できるデータを参考に作成.

このデータを学習時に使ったデータ形式に整形する関数を作成し、
学習時と推論時の差異を吸収することが目的.

推論時のデータは、test_df と prediction_df とがある.
test_df は学習データ
prediction_df が推論したい日付と選手ID が入ったからのDF 
この空のprediction_df 埋めて、predict に渡すことで評価される.

prediction_df
	date_playerId	target1	target2	target3	target4
date					
20210426	20210427_656669	0	0	0	0
...
"""

train_path = ROOT + "train_updated.csv"
csv_ro = pyarrow.csv.ReadOptions(block_size=2**25)
table = pyarrow.csv.read_csv(train_path, read_options=csv_ro)
train_df = table.to_pandas()

# %%
train = train_df[train_df.date == 20210426].reset_index(drop=True)
test_df = train.copy()
test_df.drop(["nextDayPlayerEngagement"], axis=1, inplace=True)

prediction_df = extract_data(train, col="nextDayPlayerEngagement", show=True)
prediction_df.engagementMetricsDate = pd.to_datetime(
    prediction_df.engagementMetricsDate, format="%Y-%m-%d"
)
prediction_df["date"] = prediction_df.engagementMetricsDate + dt.timedelta(days=-1)
prediction_df.date = prediction_df.date.dt.strftime("%Y%m%d")
prediction_df.engagementMetricsDate = prediction_df.engagementMetricsDate.dt.strftime(
    "%Y%m%d"
)
prediction_df["date_playerId"] = (
    prediction_df.engagementMetricsDate + "_" + prediction_df.playerId.astype(str)
)

target_cols = ["target1", "target2", "target3", "target4"]
for col in target_cols:
    prediction_df[col] = 0

use_cols = ["date", "date_playerId"] + target_cols
prediction_df = prediction_df[use_cols]
prediction_df.set_index("date", inplace=True)
# display(prediction_df)


# %%
def predict_lgb(input_x, input_id, list_n_fold=[0, 1, 2]):
    """
    fit済みのモデルがKaggle Notebook にupload ずみで読み込むだけ
    の状態を前提にする
    """
    targets = [f"target{n}" for n in list(np.arange(4) + 1)]
    df_test_pred = id_test.copy()

    for k in list_n_fold:
        for i, target in enumerate(targets):
            """
            target の種類ごとにモデルを予測
            """
            print("model load.")
            model = trained_models[k][i]
            pred = model.predict(input_x)
            df_test_pred["{}_fold{}".format(target, k)] = pred

    # 各fold の平均値
    # fold 数はlist_n_fold で決まる. 学習時と同じ= upload 済みの1target に対するcv 数
    # mean(axis=1) 同じ行のカラムのmean を計算
    for target in target_cols:
        df_test_pred[target] = df_test_pred[
            df_test_pred.columns[df_test_pred.columns.str.contains(target)]
        ].mean(axis=1)

    return df_test_pred


x_test, y_test, id_test = make_dataset_for_predict(
    input_test=test_df, input_prediction=prediction_df
)
ret = predict_lgb(input_x=x_test, input_id=id_test)

# %%
