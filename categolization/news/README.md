### Pickleをload するときの注意点
https://stackoverflow.com/questions/27732354/unable-to-load-files-using-pickle-and-multiple-modules
### 行列計算
- 要素ごとの積（`*, np.multiply`）`[a,b,c] * [d,e,f] -> [a*d, b*e, c*f]`
- ベクトル積（内積、ドット積：`np.dot, np.vdot, np.inner`）：はベクトルごとの積。つまり `[a,b,c] * [d,e,f] -> a*d + b*e + c*f`
- 行列積(`np.matmul`) : 二階行列同士の行列積。つまり　`[[a,b,c],[d,e,f]] * [[g],[h],[i]] -> [[a*g + b*i + c*k], [d*h + e*j + f*l]]`
### TF
- Useful link https://stackoverflow.com/questions/51401935/arguments-to-tensorflow-session-run-do-you-pass-operations/51402747
