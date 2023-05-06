#!/bin/bash

mkdir dst-modelselect

python3 modelselect.py -i iris.data -c -m cv > dst-modelselect/1.txt &
python3 modelselect.py -i sonar.all-data -c -m cv > dst-modelselect/2.txt &
python3 modelselect.py -i glass.data -x 0 -c -m cv > dst-modelselect/3.txt &

python3 modelselect.py -i airfoil_self_noise.dat -s '\t' -r -c -m cv > dst-modelselect/4.txt &
python3 modelselect.py -i winequality-red.csv -s ";" -e 0 -r -c -m cv > dst-modelselect/5.txt &
python3 modelselect.py -i winequality-white.csv -s ";" -e 0 -r -c -m cv > dst-modelselect/6.txt &

python3 modelselect.py -i iris.data -c -m gating > dst-modelselect/7.txt &
python3 modelselect.py -i sonar.all-data -c -m gating > dst-modelselect/8.txt &
python3 modelselect.py -i glass.data -x 0 -c -m gating > dst-modelselect/9.txt &

python3 modelselect.py -i airfoil_self_noise.dat -s '\t' -r -c -m gating > dst-modelselect/10.txt &
python3 modelselect.py -i winequality-red.csv -s ";" -e 0 -r -c -m gating > dst-modelselect/11.txt &
python3 modelselect.py -i winequality-white.csv -s ";" -e 0 -r -c -m gating > dst-modelselect/12.txt &

python3 modelselect.py -i iris.data -c -m bic > dst-modelselect/13.txt &
python3 modelselect.py -i sonar.all-data -c -m bic > dst-modelselect/14.txt &
python3 modelselect.py -i glass.data -x 0 -c -m bic > dst-modelselect/15.txt &

python3 modelselect.py -i airfoil_self_noise.dat -s '\t' -r -c -m bic > dst-modelselect/16.txt &
python3 modelselect.py -i winequality-red.csv -s ";" -e 0 -r -c -m bic > dst-modelselect/17.txt &
python3 modelselect.py -i winequality-white.csv -s ";" -e 0 -r -c -m bic > dst-modelselect/18.txt &

wait


