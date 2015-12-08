#!/bin/bash

for i in `seq 1 1 20`; do
    python dataClassifier.py -c 1 -t 5000 -s 1000 -i 5
    #python dataClassifier.py -c 2 -d faces -t 451 -s 150 -k $i
    echo ""
done
