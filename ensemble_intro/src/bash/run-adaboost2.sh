#!/bin/bash

mkdir dst-adaboost2

python3 adaboost_m1.py -i iris.data -c -b 5 > dst-adaboost2/1.txt &
python3 adaboost_m1.py -i sonar.all-data -c -b 5 > dst-adaboost2/2.txt &
python3 adaboost_m1.py -i glass.data -x 0 -c -b 5 > dst-adaboost2/3.txt &

python3 adaboost_m1.py -i iris.data -c -b 10 > dst-adaboost2/4.txt &
python3 adaboost_m1.py -i sonar.all-data -c -b 10 > dst-adaboost2/5.txt &
python3 adaboost_m1.py -i glass.data -x 0 -c -b 10 > dst-adaboost2/6.txt &

python3 adaboost_m1.py -i iris.data -c -b 20 > dst-adaboost2/7.txt &
python3 adaboost_m1.py -i sonar.all-data -c -b 20 > dst-adaboost2/8.txt &
python3 adaboost_m1.py -i glass.data -x 0 -c -b 20 > dst-adaboost2/9.txt &

python3 adaboost_rt.py -i airfoil_self_noise.dat -s '\t' -r -c -b 5 -t 0.01 > dst-adaboost2/10.txt &
python3 adaboost_rt.py -i winequality-red.csv -s ";" -e 0 -r -c -b 5 -t 0.05 > dst-adaboost2/11.txt &
python3 adaboost_rt.py -i winequality-white.csv -s ";" -e 0 -r -c -b 5 -t 0.05 > dst-adaboost2/12.txt &

python3 adaboost_rt.py -i airfoil_self_noise.dat -s '\t' -r -c -b 10 -t 0.01 > dst-adaboost2/13.txt &
python3 adaboost_rt.py -i winequality-red.csv -s ";" -e 0 -r -c -b 10 -t 0.05 > dst-adaboost2/14.txt &
python3 adaboost_rt.py -i winequality-white.csv -s ";" -e 0 -r -c -b 10 -t 0.05 > dst-adaboost2/15.txt &

python3 adaboost_rt.py -i airfoil_self_noise.dat -s '\t' -r -c -b 20 -t 0.01 > dst-adaboost2/16.txt &
python3 adaboost_rt.py -i winequality-red.csv -s ";" -e 0 -r -c -b 20 -t 0.05 > dst-adaboost2/17.txt &
python3 adaboost_rt.py -i winequality-white.csv -s ";" -e 0 -r -c -b 20 -t 0.05 > dst-adaboost2/18.txt &

python3 adaboost_r2.py -i airfoil_self_noise.dat -s '\t' -r -c -b 5 > dst-adaboost2/19.txt &
python3 adaboost_r2.py -i winequality-red.csv -s ";" -e 0 -r -c -b 5 > dst-adaboost2/20.txt &
python3 adaboost_r2.py -i winequality-white.csv -s ";" -e 0 -r -c -b 5 > dst-adaboost2/21.txt &

python3 adaboost_r2.py -i airfoil_self_noise.dat -s '\t' -r -c -b 10 > dst-adaboost2/22.txt &
python3 adaboost_r2.py -i winequality-red.csv -s ";" -e 0 -r -c -b 10 > dst-adaboost2/23.txt &
python3 adaboost_r2.py -i winequality-white.csv -s ";" -e 0 -r -c -b 10 > dst-adaboost2/24.txt &

python3 adaboost_r2.py -i airfoil_self_noise.dat -s '\t' -r -c -b 20 > dst-adaboost2/25.txt &
python3 adaboost_r2.py -i winequality-red.csv -s ";" -e 0 -r -c -b 20 > dst-adaboost2/26.txt &
python3 adaboost_r2.py -i winequality-white.csv -s ";" -e 0 -r -c -b 20 > dst-adaboost2/27.txt &

wait
