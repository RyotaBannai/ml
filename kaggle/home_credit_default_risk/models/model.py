# %%
"""
load library
"""
import gc
import os
import pickle
import warnings
from operator import itemgetter
from typing import Any, Callable, Iterable, Sequence, Tuple

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
import ydata_profiling as pdp
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from sklearn.preprocessing import (
    LabelEncoder,
    MinMaxScaler,
    OneHotEncoder,
    StandardScaler,
)
from utils import preprocess

warnings.filterwarnings("ignore")

# %%
"""
define functions
"""


def train_cv(
    input_x, input_y, input_id, fit: Callable, n_splits=5, **kwargs
) -> Tuple[pd.DataFrame, pd.DataFrame, list[list[float]]]:
    """train with cross validation"""

    # out of fold. trainで使ってないsample での予測値
    train_oof = np.zeros(len(input_x))
    metrics = []
    imp = pd.DataFrame()

    cv = list(
        StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=123).split(
            input_x, input_y
        )
    )

    # ハイパラ探索時は、cv=1 時短のため1回だけで探索
    if "is_optimize" in kwargs:
        n_splits = 1

    for k in np.arange(n_splits):
        idx_tr, idx_va = cv[k][0], cv[k][1]

        # import ipdb

        # ipdb.set_trace()
        x_tr, y_tr, id_tr = (
            input_x.loc[idx_tr, :],
            input_y.loc[idx_tr, :],
            input_id.loc[idx_tr, :],
        )
        x_va, y_va, id_va = (
            input_x.loc[idx_va, :],
            input_y.loc[idx_va, :],
            input_id.loc[idx_va, :],
        )

        print(f"{k}-fold")
        print(x_tr.shape, y_tr.shape)
        print(x_va.shape, y_va.shape)
        # train, validation, test でクラスの割合を同じにできていることを確認.
        print(
            "[accuracy] input: {:.3f}, tr: {:.3f}, va: {:.3f}".format(
                input_y.TARGET.mean(), y_tr.TARGET.mean(), y_tr.TARGET.mean()
            )
        )
        model = fit(x_tr, y_tr, x_va, y_va, **kwargs)

        fname_lgb = f"./kfold_model_pickle/model_lgb_fold{k}.pickle"
        with open(fname_lgb, "wb") as f:
            pickle.dump(model, f, protocol=4)

        y_tr_pred_prob = model.predict_proba(x_tr)[:, 1]
        y_va_pred_prob = model.predict_proba(x_va)[:, 1]
        metrix_tr = roc_auc_score(y_tr, y_tr_pred_prob)
        metrix_va = roc_auc_score(y_va, y_va_pred_prob)
        metrics.append([k, metrix_tr, metrix_va])

        print(
            "[accuracy] kfold: {}, tr_prob: {:.2f}, va_prob: {:.2f}".format(
                k, metrix_tr, metrix_va
            )
        )

        train_oof[idx_va] = y_va_pred_prob

        _imp = pd.DataFrame(
            {"nfold": k, "col": x_train.columns, "imp": model.feature_importances_}
        )
        imp = pd.concat([imp, _imp], axis=0, ignore_index=True)

    print("-" * 20, "result", "-" * 20)

    metrics = np.array(metrics)
    print("[metrices; kth, roc_auc_score tr, roc_auc_score va]:")
    print(metrics)
    print(
        "[cv] tr: {:.4f}+-{:.4f}, va: {:.4f}+-{:.4f}".format(
            metrics[:, 1].mean(),
            metrics[:, 1].std(),
            metrics[:, 2].mean(),
            metrics[:, 2].std(),
        )
    )
    # oof はユニークになっていて、index で推論結果を代入しているから、y と同じ並びで管理できている
    print("[oof; roc_auc_score] {:.4}".format(roc_auc_score(input_y, train_oof)))
    train_oof = pd.concat([input_id, pd.DataFrame({"pred": train_oof})], axis=1)

    # 重要度の平均と標準偏差を計算.
    imp = imp.groupby("col")["imp"].agg(["mean", "std"]).reset_index(drop=False)
    imp.columns = ["col", "imp", "imp_std"]
    imp.sort_values(by=["imp"], ascending=False, inplace=True)

    print("[imp]:")
    print(imp)

    return train_oof, imp, metrics


def lightgbm_hy_params(trial=None, update_params=None, is_optimize=False) -> dict:
    print("---")
    print("lightgbm params")
    print(f"{trial=}, {update_params=}, {is_optimize=}")
    print("---")

    # ハイパーパラメータ
    params_base = {
        "boosting_type": "gbdt",
        "objective": "binary",
        "metric": "auc",
        "learning_rate": 0.1,
        "num_iterations": 100000,
        "num_leaves": 16,
        "n_estimators": 100,
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


def lightgbm_fit(*args, **kwargs) -> lgb.LGBMClassifier:
    hy_params = lightgbm_hy_params(**kwargs)
    print("---")
    print("lightgbm fit hy_params")
    print(f"{hy_params=}")
    print("---")

    verbose_eval = 0
    # https://qiita.com/c60evaporator/items/2b7a2820d575e212bcf4
    callbacks = [
        # early_stopping用コールバック関数
        lgb.early_stopping(stopping_rounds=100, verbose=True),
        # コマンドライン出力用コールバック関数
        lgb.log_evaluation(verbose_eval),
    ]

    x_tr, y_tr, x_va, y_va = args
    model = lgb.LGBMClassifier(**hy_params)
    model.fit(x_tr, y_tr, eval_set=[(x_tr, y_tr), (x_va, y_va)], callbacks=callbacks)
    return model


# %%
"""
load train data
"""

input_path = "../data/application_train.csv"
pos_cash_balance_path = "../data/POS_CASH_balance.csv"
test_path = "../data/application_test.csv"
df_train = pd.read_csv(input_path)
df_train_pos = pd.read_csv(pos_cash_balance_path)


# %%
"""
前処理
"""
# データの分布解析
# pdp.ProfileReport(df_train[["TARGET", "DAYS_EMPLOYED"]])

x_train, y_train, id_train = (
    df_train,
    df_train[["TARGET"]],
    df_train[["SK_ID_CURR"]],
)

x_train.drop(
    labels=["SK_ID_CURR", "TARGET"],
    axis=1,
    inplace=True,
)
for col in x_train.columns:
    if x_train[col].dtype == "O":
        x_train[col] = x_train[col].astype("category")

x_train["DAYS_EMPLOYED"] = x_train.DAYS_EMPLOYED.replace(365243, np.nan)


# %%
"""
turning Hyperparams with Optuna
"""


# def objective(trial) -> float:
#     train_oof, imp, metrics = train_cv(
#         input_x=x_train,
#         input_y=y_train,
#         input_id=id_train,
#         fit=lightgbm_fit,
#         n_splits=5,
#         trial=trial,
#         is_optimize=True,
#     )
#     return roc_auc_score(y_train, train_oof[["pred"]])


# sampler = optuna.samplers.TPESampler(seed=123)
# study = optuna.create_study(sampler=sampler, direction="maximize")
# study.optimize(objective, n_trials=50, n_jobs=5)
# trial = study.best_trial
# print(f"acc(best)={trial.value}")
# print(trial.params)

params = {
    "num_leavels": 77,
    "min_data_in_leaf": 118,
    "min_sum_hessian_in_leaf": 4.398122146214642e-05,
    "feature_fraction": 0.6129301736043858,
    "bagging_fraction": 0.5478198788044168,
    "lambda_l1": 0.039809464414453016,
    "lambda_l2": 3.0351035127238473,
}
# %%
"""
train model
"""
train_cv(
    input_x=x_train,
    input_y=y_train,
    input_id=id_train,
    fit=lightgbm_fit,
    n_splits=5,
    # update_params=trial.params,
    update_params=params,
)


# %%

"""
load testcase and preprocess
"""
df_test = pd.read_csv(test_path)
id_test = df_test[["SK_ID_CURR"]]
df_test.drop(
    labels=[
        "SK_ID_CURR",
    ],
    axis=1,
    inplace=True,
)
for col in df_test.columns:
    if df_test[col].dtype == "O":
        df_test[col] = df_test[col].astype("category")

# %%
"""
predict testcase
"""


def predict_lgb(input_x, input_id, n_splits):
    pred = np.zeros((len(input_x), n_splits))

    for k in np.arange(n_splits):
        fname_lgb = f"./kfold_model_pickle/model_lgb_fold{k}.pickle"
        with open(fname_lgb, "rb") as f:
            model = pickle.load(f)
        pred[:, k] = model.predict_proba(input_x)[:, 1]

    # k-models の算術平均で予測値を出す.
    pred = pd.concat([input_id, pd.DataFrame({"TARGET": pred.mean(axis=1)})], axis=1)
    print("Done.")

    return pred


pred = predict_lgb(input_x=df_test, input_id=id_test, n_splits=5)

# %%
"""
create submission
"""
pred.to_csv("submission.csv", index=None)

# %%
