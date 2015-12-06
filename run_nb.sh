#!/bin/bash

for i in `seq 0.1 0.1 5`; do
    python dataClassifier.py -c 2 -t 5000 -s 1000 -k $i
    echo ""
done
