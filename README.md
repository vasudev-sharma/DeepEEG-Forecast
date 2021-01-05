# Deep Learning Based EEG toolbox
## What is does?

### 1. Predict EEG from EEG
![EEG_from EEG](https://github.com/vs74/EEG/blob/master/.static/images/EEG_from_EEG.png)
### 2. Predict EEG from Stimului
![EEG_from_Stimuli](https://github.com/vs74/EEG/blob/master/.static/images/EEG_from_Stimuli.png)
### 3. Predict Stimuli from EEG

![Stimuli_from_EEG](https://github.com/vs74/EEG/blob/master/.static/images/Stimulus_from_EEG.png)
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

## Contributing
We ❤️ contributions. Feel free to send us a PR.

1. Create an issue if there is one.
2. Fork the repo.
3. Create your feature branch (git checkout -b your-feature).
4. Add and commit your changes (git commit -am 'message').
5. Push the branch (git push origin your-feature).
6. Create a new Pull Request.

## TODO
- [ ] Implement Attention model
- [ ] Implement ESRNN model
- [ ] Deploy the model 