for i in `seq 1 30`;
do 
   export pred=$i
    python -m train
done