import json
import numpy as np

with open("experiment_log.json", "r") as read_file:
        data = read_file.readlines() 
        avg_corr = []
        print(data)
        for i in range(len(data)):
            print(data[i])
            #avg_corr.append(data[i]["Experiment_"+str(i)])
            #print(data[i][])
            #avg_corr.append(dict(data[i]["Experiment_1"])
            avg_corr.append(json.loads(data[i])["Experiment_"+str(i+1)])
        print(avg_corr)

print(np.array(avg_corr).mean())

