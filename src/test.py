import json
if __name__ == "__main__":
    for i in range(1, 65):
        if(i!=30):
            print(i, end = " ")

    data= {"Electrode_40":2.4}
    with open("corr_dat.json", "a") as write_file:
        json.dump(data, write_file)

'''  
1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61 62 63 64

0.7370166321012029
0.7985
0.69 he uniform


0.8017889453621331  
'''

