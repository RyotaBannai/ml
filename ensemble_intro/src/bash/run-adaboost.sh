#!/bin/bash

mkdir dst-adaboost

python3 adaboost.py -i sonar.all-data -c -b 5 > dst-adaboost/1.txt &
python3 adaboost.py -i sonar.all-data -c -b 10 > dst-adaboost/2.txt &
python3 adaboost.py -i sonar.all-data -c -b 20 > dst-adaboost/3.txt &

wait


