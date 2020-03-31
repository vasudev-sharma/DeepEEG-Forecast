export stimulus='2' # Whether you want to use stimulus or not 2 ---> yes, 1 ---> no
export relation='3' # What you want to predict 1--> EEG from stimulus, 2---> Stimulus from EEG, 3 --> EEG forecasting
export model_name="LSTM"
for i in -1;
do 
   export pred=$i
    
    python -m train
done