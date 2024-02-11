from operator import itemgetter
from typing import Callable, Tuple

import numpy as np
import pandas as pd


def preprocess(input_df: pd.DataFrame) -> Tuple[pd.DataFrame, list[str]]:
    """学習と推論時の入力データの前処理"""
    input_df = input_df.copy()
    input_df.loc[input_df.Cabin.isna(), "Cabin"] = "N"
    input_df["Cabin_Type"] = input_df.Cabin.map(itemgetter(0))
    input_df = input_df.astype({"Pclass": "object"})
    input_df["Age"] = input_df["Age"].fillna(input_df["Age"].mean())
    input_df["Embarked"] = input_df["Embarked"].fillna(input_df["Embarked"].mode()[0])
    num_labels = ["Fare", "Age"]
    cat_labels = ["Sex", "Embarked", "Cabin_Type", "Pclass"]
    dummied_cats = pd.get_dummies(input_df[cat_labels], drop_first=True) * 1
    input_df.drop(cat_labels + ["Cabin"], axis=1, inplace=True)
    input_df = pd.concat([input_df, dummied_cats], axis=1)
    cols = dummied_cats.columns.values.tolist() + num_labels
    return input_df, cols


def tree_predict_and_save(
    test_path, fit_cols, preprocess: Callable, model, filename="submission.csv"
) -> None:
    df_test = pd.read_csv(test_path)
    df_test, _ = preprocess(input_df=df_test)
    for col in fit_cols:
        if col not in df_test.columns:
            df_test[col] = 0
    id_test = df_test[["PassengerId"]]
    x_test = df_test[fit_cols]
    y_test_pred = model.predict_proba(x_test)[:, 1]
    survived = np.where(y_test_pred >= 0.5, 1, 0)
    df_submit = pd.DataFrame(
        {"PassengerId": id_test["PassengerId"], "survived": survived}
    )
    df_submit.to_csv(filename, index=None)


def nn_predict_and_save(
    test_path, fit_cols, preprocess: Callable, model, filename="submission.csv"
) -> None:
    df_test = pd.read_csv(test_path)
    df_test, _ = preprocess(input_df=df_test)
    for col in fit_cols:
        if col not in df_test.columns:
            df_test[col] = 0
    id_test = df_test[["PassengerId"]]
    x_test = df_test[fit_cols]
    y_test_pred = model.predict(x_test, batch_size=8, verbose=1)
    survived = np.where(y_test_pred >= 0.5, 1, 0)[:, 0]
    df_submit = pd.DataFrame(
        {"PassengerId": id_test["PassengerId"], "survived": survived}
    )
    df_submit.to_csv(filename, index=None)
