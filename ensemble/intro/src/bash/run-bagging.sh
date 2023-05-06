#!/bin/bash

mkdir dst-bagging

python3 bagging.py -i iris.data -c -t 5 > dst-bagging/1.txt &
python3 bagging.py -i sonar.all-data -c -t 5 > dst-bagging/2.txt &
python3 bagging.py -i glass.data -x 0 -c -t 5 > dst-bagging/3.txt &

python3 bagging.py -i airfoil_self_noise.dat -s '\t' -r -c -t 5 > dst-bagging/4.txt &
python3 bagging.py -i winequality-red.csv -s ";" -e 0 -r -c -t 5 > dst-bagging/5.txt &
python3 bagging.py -i winequality-white.csv -s ";" -e 0 -r -c -t 5 > dst-bagging/6.txt &

python3 bagging.py -i iris.data -c -t 10 > dst-bagging/7.txt &
python3 bagging.py -i sonar.all-data -c -t 10 > dst-bagging/8.txt &
python3 bagging.py -i glass.data -x 0 -c -t 10 > dst-bagging/9.txt &

python3 bagging.py -i airfoil_self_noise.dat -s '\t' -r -c -t 10 > dst-bagging/10.txt &
python3 bagging.py -i winequality-red.csv -s ";" -e 0 -r -c -t 10 > dst-bagging/11.txt &
python3 bagging.py -i winequality-white.csv -s ";" -e 0 -r -c -t 10 > dst-bagging/12.txt &

python3 bagging.py -i iris.data -c -t 20 > dst-bagging/13.txt &
python3 bagging.py -i sonar.all-data -c -t 20 > dst-bagging/14.txt &
python3 bagging.py -i glass.data -x 0 -c -t 20 > dst-bagging/15.txt &

python3 bagging.py -i airfoil_self_noise.dat -s '\t' -r -c -t 20 > dst-bagging/16.txt &
python3 bagging.py -i winequality-red.csv -s ";" -e 0 -r -c -t 20 > dst-bagging/17.txt &
python3 bagging.py -i winequality-white.csv -s ";" -e 0 -r -c -t 20 > dst-bagging/18.txt &

wait


