#!/bin/bash

mkdir dst-gradientboost

python3 gradientboost.py -i iris.data -c -b 5 > dst-gradientboost/1.txt &
python3 gradientboost.py -i sonar.all-data -c -b 5 > dst-gradientboost/2.txt &
python3 gradientboost.py -i glass.data -x 0 -c -b 5 > dst-gradientboost/3.txt &

python3 gradientboost.py -i airfoil_self_noise.dat -s '\t' -r -c -b 5 > dst-gradientboost/4.txt &
python3 gradientboost.py -i winequality-red.csv -s ";" -e 0 -r -c -b 5 > dst-gradientboost/5.txt &
python3 gradientboost.py -i winequality-white.csv -s ";" -e 0 -r -c -b 5 > dst-gradientboost/6.txt &

python3 gradientboost.py -i iris.data -c -b 10 > dst-gradientboost/7.txt &
python3 gradientboost.py -i sonar.all-data -c -b 10 > dst-gradientboost/8.txt &
python3 gradientboost.py -i glass.data -x 0 -c -b 10 > dst-gradientboost/9.txt &

python3 gradientboost.py -i airfoil_self_noise.dat -s '\t' -r -c -b 10 > dst-gradientboost/10.txt &
python3 gradientboost.py -i winequality-red.csv -s ";" -e 0 -r -c -b 10 > dst-gradientboost/11.txt &
python3 gradientboost.py -i winequality-white.csv -s ";" -e 0 -r -c -b 10 > dst-gradientboost/12.txt &

python3 gradientboost.py -i iris.data -c -b 20 > dst-gradientboost/13.txt &
python3 gradientboost.py -i sonar.all-data -c -b 20 > dst-gradientboost/14.txt &
python3 gradientboost.py -i glass.data -x 0 -c -b 20 > dst-gradientboost/15.txt &

python3 gradientboost.py -i airfoil_self_noise.dat -s '\t' -r -c -b 20 > dst-gradientboost/16.txt &
python3 gradientboost.py -i winequality-red.csv -s ";" -e 0 -r -c -b 20 > dst-gradientboost/17.txt &
python3 gradientboost.py -i winequality-white.csv -s ";" -e 0 -r -c -b 20 > dst-gradientboost/18.txt &

wait


