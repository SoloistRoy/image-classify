#!/bin/bash

for i in `seq 0.1 0.1 5`; do
    python dataClassifier.py -c 3 -t 1 -s 1 -i $i
    #python dataClassifier.py -c 2 -d faces -t 451 -s 150 -k $i
    echo ""
done
