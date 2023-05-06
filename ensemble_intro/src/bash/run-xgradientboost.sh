#!/bin/bash

mkdir dst-xgradientboost

python3 xgradientboost.py -i iris.data -c -b 5 > dst-xgradientboost/1.txt &
python3 xgradientboost.py -i sonar.all-data -c -b 5 > dst-xgradientboost/2.txt &
python3 xgradientboost.py -i glass.data -x 0 -c -b 5 > dst-xgradientboost/3.txt &

python3 xgradientboost.py -i airfoil_self_noise.dat -s '\t' -r -c -b 5 > dst-xgradientboost/4.txt &
python3 xgradientboost.py -i winequality-red.csv -s ";" -e 0 -r -c -b 5 > dst-xgradientboost/5.txt &
python3 xgradientboost.py -i winequality-white.csv -s ";" -e 0 -r -c -b 5 > dst-xgradientboost/6.txt &

python3 xgradientboost.py -i iris.data -c -b 10 > dst-xgradientboost/7.txt &
python3 xgradientboost.py -i sonar.all-data -c -b 10 > dst-xgradientboost/8.txt &
python3 xgradientboost.py -i glass.data -x 0 -c -b 10 > dst-xgradientboost/9.txt &

python3 xgradientboost.py -i airfoil_self_noise.dat -s '\t' -r -c -b 10 > dst-xgradientboost/10.txt &
python3 xgradientboost.py -i winequality-red.csv -s ";" -e 0 -r -c -b 10 > dst-xgradientboost/11.txt &
python3 xgradientboost.py -i winequality-white.csv -s ";" -e 0 -r -c -b 10 > dst-xgradientboost/12.txt &

python3 xgradientboost.py -i iris.data -c -b 20 > dst-xgradientboost/13.txt &
python3 xgradientboost.py -i sonar.all-data -c -b 20 > dst-xgradientboost/14.txt &
python3 xgradientboost.py -i glass.data -x 0 -c -b 20 > dst-xgradientboost/15.txt &

python3 xgradientboost.py -i airfoil_self_noise.dat -s '\t' -r -c -b 20 > dst-xgradientboost/16.txt &
python3 xgradientboost.py -i winequality-red.csv -s ";" -e 0 -r -c -b 20 > dst-xgradientboost/17.txt &
python3 xgradientboost.py -i winequality-white.csv -s ";" -e 0 -r -c -b 20 > dst-xgradientboost/18.txt &

wait


