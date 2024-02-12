# %%
import gc
import os
import pickle
import warnings
from operator import itemgetter
from typing import Any, Callable, Iterable, Sequence, Tuple

import japanize_matplotlib
import lightgbm as lgb

# 可視化
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd

# 分布確認
import ydata_profiling as pdp
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split

# 前処理
from sklearn.preprocessing import (
    LabelEncoder,
    MinMaxScaler,
    OneHotEncoder,
    StandardScaler,
)
from utils import preprocess, tree_predict_and_save

# %%

warnings.filterwarnings("ignore")


def train_cv(
    input_x, input_y, input_id, fit: Callable, n_splits=5, **kwargs
) -> Tuple[pd.DataFrame, list[float]]:
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
            "[accuracy] input: {:.3f}, tr: {:.3f}, va: {:.3f}".format(
                input_y.Survived.mean(), y_tr.Survived.mean(), y_tr.Survived.mean()
            )
        )
        model = fit(x_tr, y_tr, x_va, y_va, **kwargs)
        y_va_pred = model.predict_proba(x_va)[:, 1]
        metrix_va = accuracy_score(y_va, np.where(y_va_pred >= 0.5, 1, 0))
        metrics.append(metrix_va)
        print("[accuracy] kfold: {} va: {:.2f}".format(k, metrix_va))

        _imp = pd.DataFrame(
            {"nfold": k, "col": x_train.columns, "imp": model.feature_importances_}
        )
        imp = pd.concat([imp, _imp], axis=0, ignore_index=True)

    # 重要度の平均と標準偏差を計算.
    imp = imp.groupby("col")["imp"].agg(["mean", "std"])
    imp.columns = ["imp", "imp_std"]
    imp = imp.reset_index(drop=False)

    print("[metrices];")
    print(metrics)
    print("[imp]:")
    print(imp)

    return imp, metrics


def lightgbm_hy_params(trial=None, update_params=None, is_optimize=False) -> dict:
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
    print(hy_params)
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
input_path = "../data/train.csv"
test_path = "../data/test.csv"
df_train = pd.read_csv(input_path)
df_train, cols = preprocess(input_df=df_train)
x_train, y_train, id_train = (
    df_train[cols],
    df_train[["Survived"]],
    df_train[["PassengerId"]],
)


# %%
def objective(trial) -> float:
    _, metrics = train_cv(
        input_x=x_train,
        input_y=y_train,
        input_id=id_train,
        fit=lambda *args, **kwargs: lightgbm_fit(*args, **kwargs, is_optimize=True),
        trial=trial,
    )
    return float(np.mean(metrics))


sampler = optuna.samplers.TPESampler(seed=123)
study = optuna.create_study(sampler=sampler, direction="maximize")
study.optimize(objective, n_trials=30)
trial = study.best_trial
print(f"acc(best)={trial.value}")
print(trial.params)

# %%
pdp.ProfileReport(df_train)

# %%
# submission 作成.
x_tr, x_va, y_tr, y_va = train_test_split(
    x_train, y_train, test_size=0.2, shuffle=True, stratify=y_train, random_state=123
)
# print(x_tr.shape, y_tr.shape)
# print(x_va.shape, y_va.shape)
# print(y_train.Survived.mean(), y_tr.Survived.mean(), y_tr.Survived.mean())

model = lightgbm_fit(
    x_tr, y_tr, x_va, y_va, update_params=trial.params, is_optimize=False
)
tree_predict_and_save(
    test_path=test_path, fit_cols=cols, preprocess=preprocess, model=model
)

# %%
