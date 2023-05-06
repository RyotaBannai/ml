#!/bin/bash

mkdir dst-modelmean

python3 modelmean.py -i iris.data -c -m stacking > dst-modelmean/1.txt &
python3 modelmean.py -i sonar.all-data -c -m stacking > dst-modelmean/2.txt &
python3 modelmean.py -i glass.data -x 0 -c -m stacking > dst-modelmean/3.txt &

python3 modelmean.py -i airfoil_self_noise.dat -s '\t' -r -c -m stacking > dst-modelmean/4.txt &
python3 modelmean.py -i winequality-red.csv -s ";" -e 0 -r -c -m stacking > dst-modelmean/5.txt &
python3 modelmean.py -i winequality-white.csv -s ";" -e 0 -r -c -m stacking > dst-modelmean/6.txt &

python3 modelmean.py -i iris.data -c -m nfold > dst-modelmean/7.txt &
python3 modelmean.py -i sonar.all-data -c -m nfold > dst-modelmean/8.txt &
python3 modelmean.py -i glass.data -x 0 -c -m nfold > dst-modelmean/9.txt &

python3 modelmean.py -i airfoil_self_noise.dat -s '\t' -r -c -m nfold > dst-modelmean/10.txt &
python3 modelmean.py -i winequality-red.csv -s ";" -e 0 -r -c -m nfold > dst-modelmean/11.txt &
python3 modelmean.py -i winequality-white.csv -s ";" -e 0 -r -c -m nfold > dst-modelmean/12.txt &

python3 modelmean.py -i iris.data -c -m bic > dst-modelmean/13.txt &
python3 modelmean.py -i sonar.all-data -c -m bic > dst-modelmean/14.txt &
python3 modelmean.py -i glass.data -x 0 -c -m bic > dst-modelmean/15.txt &

python3 modelmean.py -i airfoil_self_noise.dat -s '\t' -r -c -m bic > dst-modelmean/16.txt &
python3 modelmean.py -i winequality-red.csv -s ";" -e 0 -r -c -m bic > dst-modelmean/17.txt &
python3 modelmean.py -i winequality-white.csv -s ";" -e 0 -r -c -m bic > dst-modelmean/18.txt &

wait


