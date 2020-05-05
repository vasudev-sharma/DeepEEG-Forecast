import json
import numpy as np

with open("experiment_log.json", "r") as read_file:
        data = read_file.readlines() 
        avg_corr = []
        #print(data)
        for i in range(len(data)):
            #print(data[i])
            avg_corr.append(json.loads(data[i])["Experiment_"+str(i+1)])
        #print(avg_corr)


#print(np.array(avg_corr).mean(axis = 0))



with open("models.json", "a") as write_file:
        data = np.array(avg_corr).mean(axis = 0)
        print(data)
        print(data.shape)
        json.dump(data.tolist(), write_file)
        write_file.write("\n")


with open("experiment_log.json", "w") as write_file:
        write_file.truncate(0)
        write_file.close()


