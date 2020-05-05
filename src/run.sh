export stimulus='2' # Whether you want to use stimulus or not 2 ---> no, 1 ---> yes
export input_task='3' # What you want to predict 1--> EEG from stimulus, 2---> Stimulus from EEG, 3 --> EEG forecasting
export model_name="LSTM"
export horizon=160
export training="True"
export forecasting_self="True" 
export MIMO_output="True"
#set pred to -1 if you want to predict future value of EEG using their past value and if you want to want to preidict single electrode future value from their past value declare a sequence in the for loop like this `seq 1 30`
for i in 30;
do 
   export pred=$i
    python -m train
done