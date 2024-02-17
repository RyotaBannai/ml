import numpy as np
import pandas as pd

# %%
"""
json をパースしてテーブルを作成するスクリプト
train のカラムに対して適用
"""


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
