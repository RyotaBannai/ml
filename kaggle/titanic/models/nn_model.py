# %%
import gc
import os
import pickle
import warnings
from operator import itemgetter
from typing import Any, Callable, Iterable, Sequence, Tuple

import japanize_matplotlib
import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from sklearn.preprocessing import (
    LabelEncoder,
    MinMaxScaler,
    OneHotEncoder,
    StandardScaler,
)
from tensorflow.keras.callbacks import (
    EarlyStopping,
    LearningRateScheduler,
    ModelCheckpoint,
    ReduceLROnPlateau,
)
from tensorflow.keras.layers import (
    BatchNormalization,
    Concatenate,
    Dense,
    Dropout,
    Embedding,
    Flatten,
    Input,
)
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import SGD, Adam
from utils import nn_predict_and_save, preprocess


# %%
def seed_everything(seed):
    import random

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    session_conf = tf.compat.v1.ConfigProto(
        intra_op_parallelism_threads=1,
        inter_op_parallelism_threads=1,
    )
    sess = tf.compat.v1.Session(
        graph=tf.compat.v1.get_default_graph(), config=session_conf
    )
    tf.compat.v1.keras.backend.set_session(sess)


# %%
input_path = "../data/train.csv"
test_path = "../data/test.csv"
df_train = pd.read_csv(input_path)

x_train, y_train, id_train = (
    df_train,
    df_train[["Survived"]],
    df_train[["PassengerId"]],
)


def nn_preprocess(input_df: pd.DataFrame) -> pd.DataFrame:
    input_df, cols = preprocess(input_df=input_df)
    input_df = input_df[cols]
    mms = MinMaxScaler(feature_range=(0, 1))
    mms.fit(input_df)
    input_df[input_df.columns] = mms.transform(input_df)

    return input_df


x_train = nn_preprocess(input_df=x_train)

x_tr, x_va, y_tr, y_va = train_test_split(
    x_train, y_train, test_size=0.2, shuffle=True, stratify=y_train, random_state=123
)


# %%
def create_nn_model():
    """
    features_num x 10x10x5 x 1 のnn モデル
    """
    input_num = Input(shape=(len(x_train.columns),))

    x_num = Dense(10, activation="relu")(input_num)
    x_num = BatchNormalization()(x_num)
    x_num = Dropout(0.3)(x_num)

    x_num = Dense(10, activation="relu")(x_num)
    x_num = BatchNormalization()(x_num)
    x_num = Dropout(0.2)(x_num)

    x_num = Dense(5, activation="relu")(x_num)
    x_num = BatchNormalization()(x_num)
    x_num = Dropout(0.1)(x_num)

    out = Dense(1, activation="sigmoid")(x_num)

    model = Model(inputs=input_num, outputs=out)

    model.compile(
        optimizer="Adam", loss="binary_crossentropy", metrics=["binary_crossentropy"]
    )

    return model


# %%
seed_everything(seed=123)
model = create_nn_model()
model.summary()
model.fit(
    x=x_tr,
    y=y_tr,
    validation_data=(x_va, y_va),
    batch_size=8,
    epochs=10000,
    callbacks=[
        ModelCheckpoint(
            filepath="model_keras.h5",
            monitor="val_loss",
            mode="min",
            verbose=1,
            save_best_only=True,
            save_weights_only=True,
        ),
        EarlyStopping(
            monitor="val_loss",
            mode="min",
            min_delta=0,
            patience=10,
            verbose=1,
            restore_best_weights=True,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            mode="min",
            factor=0.1,
            patience=5,
            verbose=1,
        ),
    ],
    verbose=1,
)

# %%

# model.predict(x_va, batch_size=8, verbose=1)
fit_cols = x_train.columns.values.tolist()
nn_predict_and_save(
    test_path=test_path, fit_cols=fit_cols, preprocess=preprocess, model=model
)

# %%
