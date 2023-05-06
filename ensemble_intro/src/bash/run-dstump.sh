#!/bin/bash

mkdir dst-dstump

python3 dstump.py -i iris.data -c -m gini > dst-dstump/1.txt &
python3 dstump.py -i sonar.all-data -c -m gini > dst-dstump/2.txt &
python3 dstump.py -i glass.data -x 0 -c -m gini > dst-dstump/3.txt &

python3 dstump.py -i iris.data -c -m infgain > dst-dstump/4.txt &
python3 dstump.py -i sonar.all-data -c -m infgain > dst-dstump/5.txt &
python3 dstump.py -i glass.data -x 0 -c -m infgain > dst-dstump/6.txt &

python3 dstump.py -i airfoil_self_noise.dat -s '\t' -r -c -m div > dst-dstump/7.txt &
python3 dstump.py -i winequality-red.csv -s ";" -e 0 -r -c -m div > dst-dstump/8.txt &
python3 dstump.py -i winequality-white.csv -s ";" -e 0 -r -c -m div > dst-dstump/9.txt &

wait


