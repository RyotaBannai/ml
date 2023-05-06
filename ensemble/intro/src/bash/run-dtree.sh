#!/bin/bash

mkdir dst-dtree

python3 dtree.py -i iris.data -c -d 3 > dst-dtree/1.txt &
python3 dtree.py -i sonar.all-data -c -d 3 > dst-dtree/2.txt &
python3 dtree.py -i glass.data -x 0 -c -d 3 > dst-dtree/3.txt &

python3 dtree.py -i airfoil_self_noise.dat -s '\t' -r -c -d 3 > dst-dtree/4.txt &
python3 dtree.py -i winequality-red.csv -s ";" -e 0 -r -c -d 3 > dst-dtree/5.txt &
python3 dtree.py -i winequality-white.csv -s ";" -e 0 -r -c -d 3 > dst-dtree/6.txt &

python3 dtree.py -i iris.data -c -d 5 > dst-dtree/7.txt &
python3 dtree.py -i sonar.all-data -c -d 5 > dst-dtree/8.txt &
python3 dtree.py -i glass.data -x 0 -c -d 5 > dst-dtree/9.txt &

python3 dtree.py -i airfoil_self_noise.dat -s '\t' -r -c -d 5 > dst-dtree/10.txt &
python3 dtree.py -i winequality-red.csv -s ";" -e 0 -r -c -d 5 > dst-dtree/11.txt &
python3 dtree.py -i winequality-white.csv -s ";" -e 0 -r -c -d 5 > dst-dtree/12.txt &

python3 dtree.py -i iris.data -c -d 7 > dst-dtree/13.txt &
python3 dtree.py -i sonar.all-data -c -d 7 > dst-dtree/14.txt &
python3 dtree.py -i glass.data -x 0 -c -d 7 > dst-dtree/15.txt &

python3 dtree.py -i airfoil_self_noise.dat -s '\t' -r -c -d 7 > dst-dtree/16.txt &
python3 dtree.py -i winequality-red.csv -s ";" -e 0 -r -c -d 7 > dst-dtree/17.txt &
python3 dtree.py -i winequality-white.csv -s ";" -e 0 -r -c -d 7 > dst-dtree/18.txt &

wait


