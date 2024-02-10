# %%
import gc
import os
import pickle
import warnings
from typing import Any, Callable, Tuple

import lightgbm as lgb

# 可視化
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 分布確認
import pandas_profiling as pdp
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split

# 前処理
# 前処理
from sklearn.preprocessing import (
    LabelEncoder,
    MinMaxScaler,
    OneHotEncoder,
    StandardScaler,
)

warnings.filterwarnings("ignore")

import japanize_matplotlib


# %%
def train_cv(input_x, input_y, input_id, fit: Callable, n_splits=5):
    """train with cross validation"""
    metrics = []
    imp = pd.DataFrame()

    cv = list(
        StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=123).split(
            input_x, input_y
        )
    )
    for k in np.arange(n_splits):
        idx_tr, idx_va = cv[k][0], cv[k][1]
        x_tr, y_tr = input_x.loc[idx_tr, :], input_y.loc[idx_tr, :]
        x_va, y_va = input_x.loc[idx_va, :], input_y.loc[idx_va, :]

        print(f"{k}-fold")
        print(x_tr.shape, y_tr.shape)
        print(x_va.shape, y_va.shape)
        # train, validation, test でクラスの割合を同じにできていることを確認.
        print(
            "[accuracy] tr: {:.3f}, va: {:.3f}, va: {:.3f}".format(
                input_y.Survived.mean(), y_tr.Survived.mean(), y_tr.Survived.mean()
            )
        )
        model = fit(x_tr, y_tr, x_va, y_va)

        y_tr_pred = model.predict(x_tr)
        y_va_pred = model.predict(x_va)
        metrix_tr = accuracy_score(y_tr, y_tr_pred)
        metrix_va = accuracy_score(y_va, y_va_pred)
        metrics.append([k, metrix_tr, metrix_va])
        print("[accuracy] tr: {:.2f}, va: {:.2f}".format(metrix_tr, metrix_tr))

        _imp = pd.DataFrame(
            {"col": x_train.columns, "imp": model.feature_importances_, "nfold": k}
        )
        imp = pd.concat([imp, _imp], axis=0, ignore_index=True)

    metrics = np.array(metrics)
    # 重要度の平均と標準偏差を計算.
    imp = imp.groupby("col")["imp"].agg(["mean", "std"])
    imp.columns = ["imp", "imp_std"]
    imp = imp.reset_index(drop=False)

    print("[metrices];")
    print(metrics)
    print("[imp]:")
    print(imp)

    return imp, metrics


def lightgbm_fit(*args) -> lgb.LGBMClassifier:
    # ハイパーパラメータ
    params = {
        "boosting_type": "gbdt",
        "objective": "binary",
        "metric": "auc",
        "learning_rate": 0.1,
        "num_leaves": 16,
        "n_estimators": 100000,
        "random_state": 123,
        "importance_type": "gain",
    }
    verbose_eval = 0
    # https://qiita.com/c60evaporator/items/2b7a2820d575e212bcf4
    callbacks = [
        # early_stopping用コールバック関数
        lgb.early_stopping(stopping_rounds=10000, verbose=True),
        # コマンドライン出力用コールバック関数
        lgb.log_evaluation(verbose_eval),
    ]

    x_tr, y_tr, x_va, y_va = args
    model = lgb.LGBMClassifier(**params)
    model.fit(x_tr, y_tr, eval_set=[(x_tr, y_tr), (x_va, y_va)], callbacks=callbacks)
    return model


def preprocess(input_df: pd.DataFrame) -> Tuple[pd.DataFrame, list[str]]:
    """学習と推論時の入力データの前処理"""
    input_df = input_df.copy()
    input_df.loc[input_df.Cabin.isna(), "Cabin"] = "N"
    input_df["Cabin_Type"] = input_df.Cabin.map(lambda x: x[0])
    in_nums = ["Pclass", "Fare", "Age"]
    in_cats = ["Sex", "Embarked", "Cabin_Type"]
    dummied_cats = (
        pd.get_dummies(
            input_df[in_cats],
            dummy_na=False,
            drop_first=False,
            # dummy_na=True, drop_first=True # この代わりに、na をdrop して、カテゴリ変数を全てkeep してマルチコ回避
        )
        * 1
    )
    input_df.drop(in_cats + ["Cabin"], axis=1, inplace=True)
    input_df = pd.concat([input_df, dummied_cats], axis=1)

    all_dummied_cats = [
        x for x in input_df.columns if "Cabin" in x or "Embarked" in x or "Sex" in x
    ]

    return input_df, all_dummied_cats + in_nums


def predict_and_save(
    test_path, fit_cols, preprocess: Callable, model, filename="submission.csv"
):

    df_test = pd.read_csv(test_path)
    df_test, _ = preprocess(input_df=df_test)
    for col in fit_cols:
        if col not in df_test.columns:
            df_test[col] = 0
    id_test = df_test[["PassengerId"]]
    x_test = df_test[fit_cols]
    y_test_pred = model.predict(x_test)
    df_submit = pd.DataFrame(
        {"PassengerId": id_test["PassengerId"], "survived": y_test_pred}
    )
    df_submit.to_csv(filename, index=None)


# %%
input_path = "../data/train.csv"
test_path = "../data/test.csv"

# %%
df_train = pd.read_csv(input_path)
df_train, cols = preprocess(input_df=df_train)
x_train, y_train, id_train = (
    df_train[cols],
    df_train[["Survived"]],
    df_train[["PassengerId"]],
)


# %%
# # CV
# imp, metrices = train_cv(
#     input_x=x_train, input_y=y_train, input_id=id_train, fit=lightgbm_fit
# )

# %%
x_tr, x_va, y_tr, y_va = train_test_split(
    x_train, y_train, test_size=0.2, shuffle=True, stratify=y_train, random_state=123
)
print(x_tr.shape, y_tr.shape)
print(x_va.shape, y_va.shape)
print(y_train.Survived.mean(), y_tr.Survived.mean(), y_tr.Survived.mean())

# %%
model = lightgbm_fit(x_tr, y_tr, x_va, y_va)
predict_and_save(test_path=test_path, fit_cols=cols, preprocess=preprocess, model=model)

# %%
pdp.ProfileReport(df_train)
