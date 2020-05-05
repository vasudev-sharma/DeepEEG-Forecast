#!/bin/bash
echo "This script is about to run another script."

for i in `seq 1 1`
do
    export experiment_no=$i
    sh ./run.sh
    echo "This script has just run another script."
    echo "Experiment $i has been run"
done
echo "10 experiments have been run"
python -m evalute_experiments.py
