export stimulus='2' # Whether you want to use stimulus or not 2 ---> no, 1 ---> yes
export input_task='2' # What you want to predict 1--> EEG from stimulus, 2---> Stimulus from EEG, 3 --> EEG forecasting
export model_name="LSTM_autoencoder"
export horizon=160
export training="True"
export forecasting_self="True" 
export MIMO_output="True"
export load_checkpoint=""
#set pred to -1 if you want to predict future value of EEG using their past value and if you want to want to preidict single electrode future value from their past value declare a sequence in the for loop like this `seq 1 30`
for i in  62;
    do 
    export pred=$i
    #Perform multiple experiments
    for j in  in `seq 1 5`;
        do
            export experiment_no=$j
            echo "Experiment $j has been run"
            python -m train
        done

    echo "All experiments have been run"
    python -m evalute_experiments.py
    done


    
    