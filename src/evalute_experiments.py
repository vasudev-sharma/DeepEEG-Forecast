import json

'''
with open("experiment_log.json", "r") as read_file:
        data = read_file
        print(data)
        avg_corr = []
        for i in range(10):
            avg_corr.append(data["Experiment_"+str(i)])
        print(avg_corr)
'''

data = []
with open('experiment_log.json') as f:
    for i, line in enumerate(f, start = 1):
        print(json.loads(line))
        data.append(json.loads(line)["Experiment_"+str(i)])
print(data)