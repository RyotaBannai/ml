# %%
import gc
import os
import pickle
import warnings

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
df_train = pd.read_csv("./data/train.csv")
df_test = pd.read_csv("./data/test.csv")
x_train, y_train, id_train = (
    df_train[["Pclass", "Fare"]],
    df_train[["Survived"]],
    df_train[["PassengerId"]],
)

# %%
# hold-out pattern
x_tr, x_va, y_tr, y_va = train_test_split(
    x_train, y_train, test_size=0.2, shuffle=True, stratify=y_train, random_state=123
)

print(x_tr.shape, y_tr.shape)
print(x_va.shape, y_va.shape)
# train, validation, test でクラスの割合を同じにできていることを確認.
print(y_train.Survived.mean(), y_tr.Survived.mean(), y_tr.Survived.mean())

# ハイパラーパラメータ
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
callbacks = [
    # early_stopping用コールバック関数
    lgb.early_stopping(stopping_rounds=10000, verbose=True),
    # コマンドライン出力用コールバック関数
    lgb.log_evaluation(verbose_eval),
]


model = lgb.LGBMClassifier(**params)
model.fit(x_tr, y_tr, eval_set=[(x_tr, y_tr), (x_va, y_va)], callbacks=callbacks)

# %%
y_tr_pred = model.predict(x_tr)
y_va_pred = model.predict(x_va)
metrix_tr = accuracy_score(y_tr, y_tr_pred)
metrix_va = accuracy_score(y_va, y_va_pred)
print("[accuracy] tr: {:.2f}, va: {:.2f}".format(metrix_tr, metrix_tr))

# %%
imp = pd.DataFrame({"col": x_train.columns, "imp": model.feature_importances_})
imp.sort_values(by="imp", ascending=False, ignore_index=True)


# %%
# cross validation pattern
metrics = []
imp = pd.DataFrame()

n_splits = 5
cv = list(
    StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=123).split(
        x_train, y_train
    )
)
for nfold in np.arange(n_splits):
    idx_tr, idx_va = cv[nfold][0], cv[nfold][1]
    x_tr, y_tr = x_train.loc[idx_tr, :], y_train.loc[idx_tr, :]
    x_va, y_va = x_train.loc[idx_va, :], y_train.loc[idx_va, :]
    print(x_tr.shape, y_tr.shape)
    print(x_va.shape, y_va.shape)
    # train, validation, test でクラスの割合を同じにできていることを確認.
    print(
        "[accuracy] tr: {:.3f}, va: {:.3f}, va: {:.3f}".format(
            y_train.Survived.mean(), y_tr.Survived.mean(), y_tr.Survived.mean()
        )
    )
    model = lgb.LGBMClassifier(**params)
    model.fit(x_tr, y_tr, eval_set=[(x_tr, y_tr), (x_va, y_va)], callbacks=callbacks)

    y_tr_pred = model.predict(x_tr)
    y_va_pred = model.predict(x_va)
    metrix_tr = accuracy_score(y_tr, y_tr_pred)
    metrix_va = accuracy_score(y_va, y_va_pred)
    print("[accuracy] tr: {:.2f}, va: {:.2f}".format(metrix_tr, metrix_tr))
    metrics.append([nfold, metrix_tr, metrix_va])

    _imp = pd.DataFrame(
        {"col": x_train.columns, "imp": model.feature_importances_, "nfold": nfold}
    )
    imp = pd.concat([imp, _imp], axis=0, ignore_index=True)

# %%

metrics = np.array(metrics)
print(metrics)

# 重要度の平均と標準偏差を計算.
imp = imp.groupby("col")["imp"].agg(["mean", "std"])
imp.columns = ["imp", "imp_std"]
imp = imp.reset_index(drop=False)
print(imp)

# %%
x_test = df_test[["Pclass", "Fare"]]
id_test = df_test[["PassengerId"]]
y_test_pred = model.predict(x_test)
df_submit = pd.DataFrame(
    {"PassengerId": id_test["PassengerId"], "survived": y_test_pred}
)
print(df_submit.head(5))
df_submit.to_csv("submission_baseline.csv", index=None)
# %%
