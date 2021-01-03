# Deep Learning Based EEG toolbox
## What is does?

### 1. Predict EEG from EEG
### 2. Predict EEG from Stimului
### 3. Predict Stimuli from EEG

## Install
1. Clone the repo
~~~
git clone https://github.com/vs74/EEG
cd EEG
~~~
2. Set up Virtual Environment
```
virtualenv eeg_env --python=3.6
source eeg_env/bin/activate
```

3. Now, in the terminal run
```
bash install_files.sh
```

## Data
5 subjects EEG data downsampled at 160Hz <br>

Each subject has 
- 64 channels
- 192 trials
- 840 time points (5.25s) 

Data can be found here - https://drive.google.com/drive/folders/1_gV6t5f2FDWo8OYTcAxsDxz4yhRcYKU0?usp=sharing

## TODO
- [ ] Implement Attention model
- [ ] Implement ESRNN model
- [ ] Deploy the model 