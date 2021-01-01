#!/bin/bash
pip install virtualenv
virtualenv eeg_env --python=3.6
source eeg_env/bin/activate
pip install -r requirements.txt
mkdir input
mkdir images
cd input/ 
wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=1lA9mQ-7jbSP96gsdWV5eSWxyCL2_Ov52' -O 1filtered.mat
cd ../src/
mkdir experiment
wandb login

