#!/bin/bash
git checkout CNN
pip install virtualenv
virtualenv eeg_test --python=3.6
source eeg_test/bin/activate
pip install -r requirements.txt
mkdir input
mkdir images
cd input/ 
wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=1Ec4P523hO2bSWKqc2YaEVSheZXnU6Kc4' -O 1filtered.mat
cd ../src/

