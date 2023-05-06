"""
DecisionTreeRegressor playground
reference:https://towardsdatascience.com/decision-tree-regressor-a-visual-guide-with-scikit-learn-2aa9e01f5d7f

指定できるのは、criterion のみ
DecisionTreeRegressor はノードにあるサンプルデータの教師データやその平均などからcriterion によりデータを分割していく.
predict する時は、入力値を分割した時の条件をもとに葉を決定する.
葉における推論値は`学習時のサンプルデータの平均値`

reference:https://hacarus.github.io/interpretable-ml-book-ja/tree.html#%E6%B1%BA%E5%AE%9A%E6%9C%A8%E3%81%AE%E8%A7%A3%E9%87%88
"""
# %%
# Import the required libraries
import matplotlib.pyplot as plt
import numpy as np
from sklearn import tree
from sklearn.tree import DecisionTreeRegressor

# Define the dataset
X = np.array([[1], [3], [4], [7], [9], [10], [11], [13], [14], [16]])
y = np.array([3, 4, 3, 15, 17, 15, 18, 7, 3, 4])

# matplotlib 同じグラフに複数plot
# https://qiita.com/trami/items/b501abe7667e55ab2c9f
# line plot https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.plot.html
fig, ax = plt.subplots()
ax.grid()

ax.scatter(X, y, color="blue", label="Data")

model = DecisionTreeRegressor(max_depth=2)
model.fit(X, y)
# Generate predictions for a sequence of x values
x_seq = np.arange(0, 17, 0.1).reshape(-1, 1)
y_pred = model.predict(x_seq)
ax.plot(x_seq, y_pred, "r", label="Predict")
ax.legend()  # 最後に呼ぶ

plt.show()
tree.plot_tree(model, fontsize=8)
print(model.get_params())
# %%
