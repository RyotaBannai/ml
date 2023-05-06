#!/bin/bash

mkdir dst-randomforest

python3 randomforest.py -i iris.data -c -t 5 > dst-randomforest/1.txt &
python3 randomforest.py -i sonar.all-data -c -t 5 > dst-randomforest/2.txt &
python3 randomforest.py -i glass.data -x 0 -c -t 5 > dst-randomforest/3.txt &

python3 randomforest.py -i airfoil_self_noise.dat -s '\t' -r -c -t 5 > dst-randomforest/4.txt &
python3 randomforest.py -i winequality-red.csv -s ";" -e 0 -r -c -t 5 > dst-randomforest/5.txt &
python3 randomforest.py -i winequality-white.csv -s ";" -e 0 -r -c -t 5 > dst-randomforest/6.txt &

python3 randomforest.py -i iris.data -c -t 10 > dst-randomforest/7.txt &
python3 randomforest.py -i sonar.all-data -c -t 10 > dst-randomforest/8.txt &
python3 randomforest.py -i glass.data -x 0 -c -t 10 > dst-randomforest/9.txt &

python3 randomforest.py -i airfoil_self_noise.dat -s '\t' -r -c -t 10 > dst-randomforest/10.txt &
python3 randomforest.py -i winequality-red.csv -s ";" -e 0 -r -c -t 10 > dst-randomforest/11.txt &
python3 randomforest.py -i winequality-white.csv -s ";" -e 0 -r -c -t 10 > dst-randomforest/12.txt &

python3 randomforest.py -i iris.data -c -t 20 > dst-randomforest/13.txt &
python3 randomforest.py -i sonar.all-data -c -t 20 > dst-randomforest/14.txt &
python3 randomforest.py -i glass.data -x 0 -c -t 20 > dst-randomforest/15.txt &

python3 randomforest.py -i airfoil_self_noise.dat -s '\t' -r -c -t 20 > dst-randomforest/16.txt &
python3 randomforest.py -i winequality-red.csv -s ";" -e 0 -r -c -t 20 > dst-randomforest/17.txt &
python3 randomforest.py -i winequality-white.csv -s ";" -e 0 -r -c -t 20 > dst-randomforest/18.txt &

wait


