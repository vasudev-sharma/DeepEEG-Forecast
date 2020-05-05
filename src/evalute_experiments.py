import json
import numpy as np
from utils import compare_models

'''
with open("experiment_log.json", "r") as read_file:
        data = read_file.readlines() 
        avg_corr = []
        #print(data)
        for i in range(len(data)):
            #print(data[i])
            avg_corr.append(json.loads(data[i])["Experiment_"+str(i+1)])
        #print(avg_corr)


#print(np.array(avg_corr).mean(axis = 0))


#Average the experiments predictions and store them in models.json file
with open("models.json", "a") as write_file:
        mean = np.array(avg_corr).mean(axis = 0)
        std = np.array(avg_corr).std(axis = 0)
        print(mean)
        print(mean.shape)
        json.dump(mean.tolist(), write_file)
        write_file.write("\n")
        json.dump(std.tolist(), write_file)
        write_file.write("\n")


#Clear the Experiment Log
with open("experiment_log.json", "w") as write_file:
        write_file.truncate(0)
        write_file.close()
'''

compare_models()

