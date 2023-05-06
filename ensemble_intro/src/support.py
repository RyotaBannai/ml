import numpy as np


# クラス名カラムから教師データを作成
def clz_to_prob(clz):
    l = sorted(list(set(clz)))  # ユニークなクラスを用意
    m = [l.index(c) for c in clz]  # クラス名をindex 名に変換
    z = np.zeros((len(clz), len(l)))  # クラス数 x データ数の配列を作成
    for i, j in enumerate(m):  # i 番目の配列の正解(index)に正解ラベル1.0 を割り当てる
        z[i, j] = 1.0
    return z, list(map(str, l))


def prob_to_clz(prob, cl):
    i = prob.argmax(axis=1)
    return [cl[z] for z in i]


def get_base_args():
    import argparse

    ps = argparse.ArgumentParser(description="ML Test")
    ps.add_argument("--input", "-i", help="Training file")
    ps.add_argument("--separator", "-s", default=",", help="CSV separator")
    ps.add_argument("--header", "-e", type=int, default=None, help="CSV header")
    ps.add_argument("--indexcol", "-x", type=int, default=None, help="CSV index_col")
    ps.add_argument("--regression", "-r", action="store_true", help="Regression")
    ps.add_argument("--crossvalidate", "-c", action="store_true", help="Use Cross Validation")
    return ps


def report_classifier(plf, x, y, clz, cv=True):
    import warnings

    from sklearn.exceptions import UndefinedMetricWarning
    from sklearn.metrics import accuracy_score, classification_report, f1_score
    from sklearn.model_selection import KFold

    if not cv:
        plf.fit(x, y)
        print("Model:")
        print(str(plf))
        z = plf.predict(x)
        z = z.argmax(axis=1)
        y = y.argmax(axis=1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UndefinedMetricWarning)
            rp = classification_report(y, z, target_names=clz)
        print("Train Score:")
        print(rp)
    else:
        kf = KFold(n_splits=10, random_state=1, shuffle=True)
        f1 = []
        pr = []
        n = []
        for train_index, test_index in kf.split(x):
            x_train, x_test = x[train_index], x[test_index]
            y_train, y_test = y[train_index], y[test_index]
            plf.fit(x_train, y_train)
            z = plf.predict(x_test)
            z = z.argmax(axis=1)
            y_test = y_test.argmax(axis=1)
            f1.append(f1_score(y_test, z, average="weighted"))  # f1 スコア: Precision とRecall の調和平均p21
            pr.append(accuracy_score(y_test, z))
            n.append(len(x_test) / len(x))
        print("CV Score:")
        print("  F1 Score = %f" % (np.average(f1, weights=n)))
        print("  Accuracy Score = %f" % (np.average(pr, weights=n)))


def report_regressor(plf, x, y, cv=True):
    from sklearn.metrics import (
        explained_variance_score,
        mean_absolute_error,
        mean_squared_error,
        r2_score,
    )
    from sklearn.model_selection import KFold

    if not cv:
        plf.fit(x, y)
        print("Model:")
        print(str(plf))
        z = plf.predict(x)
        print("Train Score:")
        rp = r2_score(y, z)
        print("  R2 Score: %f" % rp)
        rp = explained_variance_score(y, z)
        print("  Explained Variance Score: %f" % rp)
        rp = mean_absolute_error(y, z)
        print("  Mean Absolute Error: %f" % rp)
        rp = mean_squared_error(y, z)
        print("  Mean Squared Error: %f" % rp)
    else:
        kf = KFold(n_splits=10, random_state=1, shuffle=True)
        r2 = []
        ma = []
        n = []
        for train_index, test_index in kf.split(x):
            x_train, x_test = x[train_index], x[test_index]
            y_train, y_test = y[train_index], y[test_index]
            plf.fit(x_train, y_train)
            z = plf.predict(x_test)
            r2.append(r2_score(y_test, z))
            ma.append(mean_squared_error(y_test, z))
            n.append(len(x_test) / len(x))
        print("CV Score:")
        print("  R2 Score = %f" % (np.average(r2, weights=n)))
        print("  Mean Squared Error = %f" % (np.average(ma, weights=n)))
