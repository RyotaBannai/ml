#!/bin/bash

mkdir dst-pruning

python3 pruning.py -i iris.data -c -d 7 -p reduce > dst-pruning/1.txt &
python3 pruning.py -i sonar.all-data -c -d 7 -p reduce > dst-pruning/2.txt &
python3 pruning.py -i glass.data -x 0 -c -d 7 -p reduce > dst-pruning/3.txt &

python3 pruning.py -i airfoil_self_noise.dat -s '\t' -r -c -d 7 -p reduce > dst-pruning/4.txt &
python3 pruning.py -i winequality-red.csv -s ";" -e 0 -r -c -d 7 -p reduce > dst-pruning/5.txt &
python3 pruning.py -i winequality-white.csv -s ";" -e 0 -r -c -d 7 -p reduce > dst-pruning/6.txt &

python3 pruning.py -i iris.data -c -d 7 -t -p reduce > dst-pruning/7.txt &
python3 pruning.py -i sonar.all-data -c -d 7 -t -p reduce > dst-pruning/8.txt &
python3 pruning.py -i glass.data -x 0 -c -d 7 -t -p reduce > dst-pruning/9.txt &

python3 pruning.py -i airfoil_self_noise.dat -s '\t' -r -c -d 7 -t -p reduce > dst-pruning/10.txt &
python3 pruning.py -i winequality-red.csv -s ";" -e 0 -r -c -d 7 -t -p reduce > dst-pruning/11.txt &
python3 pruning.py -i winequality-white.csv -s ";" -e 0 -r -c -d 7 -t -p reduce > dst-pruning/12.txt &

python3 pruning.py -i iris.data -c -d 7 -p critical > dst-pruning/13.txt &
python3 pruning.py -i sonar.all-data -c -d 7 -p critical > dst-pruning/14.txt &
python3 pruning.py -i glass.data -x 0 -c -d 7 -p critical > dst-pruning/15.txt &

python3 pruning.py -i airfoil_self_noise.dat -s '\t' -r -c -d 7 -p critical > dst-pruning/16.txt &
python3 pruning.py -i winequality-red.csv -s ";" -e 0 -r -c -d 7 -p critical > dst-pruning/17.txt &
python3 pruning.py -i winequality-white.csv -s ";" -e 0 -r -c -d 7 -p critical > dst-pruning/18.txt &

wait


