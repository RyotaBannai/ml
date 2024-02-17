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

import numpy as np
import pandas as pd
import pyarrow
import pyarrow.csv
from utils import extract_data

warnings.filterwarnings("ignore")
pd.options.display.float_format = "{:10.4f}".format


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

    df_players = pd.read_csv("../data/players.csv")
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

train_path = "../data/train_updated.csv"
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
        for target in targets:
            """
            target の種類ごとにモデルを予測
            """
            fname_lgb = f"./kfold_model_pickle/model_lgb_fold{k}_{target}.pickle"

            print("model load.")
            with open(fname_lgb, "rb") as f:
                model = pickle.load(f)
            pred = model.predict(input_x)
            df_test_pred["{}_fold{}".format(target, k)] = pred

    # 各fold の平均値
    # fold 数はlist_n_fold で決まる. 学習時と同じ= upload 済みの1target に対するcv 数
    # mean(axis=1) 同じ行のカラムのmean を計算
    for target in target_cols:
        df_test_pred[target] = df_test_pred[
            df_test_pred.columns[df_test_pred.columns.str.contains(target)]
        ].mean(axis=1)


x_test, y_test, id_test = make_dataset_for_predict(
    input_test=test_df, input_prediction=prediction_df
)
ret = predict_lgb(input_x=x_test, input_id=id_test)
