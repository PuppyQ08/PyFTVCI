import numpy as np
import csv
reslt = np.load("../data/thermoresult_Lambda0.npy")
filename = "../data/thermoresults_Lambda0.csv"
with open(filename,'w') as csvfile:
    csvwriter = csv.writer(csvfile)
    for i in range(4):
        for ii in range(7):
            csvwriter.writerow([reslt[ii,0,i],reslt[ii,1,i],reslt[ii,2,i]])
