import numpy as np 

import sys
import os

import time 
from datetime import datetime

#Necessary to ensure interpreter sees the current directory 

# Get the current directory of the script
current_dir = os.path.dirname(os.path.realpath(__file__))

# Add the directory to sys.path
sys.path.append(current_dir)

import pso 


# Convert the timestamp to a datetime object
datetime_object = datetime.fromtimestamp(time.time())

# Format the datetime object as a string
formatted_datetime = datetime_object.strftime('%Y-%m-%d %H:%M:%S')


print("start: ",formatted_datetime)



#for n in [50,100,200]:
#for n in [200]:
for n in [50,100]:
    filename1 = "C:\\Users\\Nefeli\\Desktop\\opti_project2\\pso_lbest_results\\vecs_pso_lbest_N_"+str(n)+".txt"
    filename2 = "C:\\Users\\Nefeli\\Desktop\\opti_project2\\pso_lbest_results\\sol_pso_lbest_N_"+str(n)+".txt"
    with open(filename1, 'w+') as file1, open(filename2, 'w+') as file2:
   
        for i in range(0,25):
            #print(i)
            results = pso.pso_main([0.0,1.0], n, 57, 0.729, 2.05,2.05,0.005,5)
        
            file1.write(f"{i + 1}")
            file1.write(',')
            file1.write(str(results[0]))
            file1.write(',')
            file1.write(str(results[1]))
            file1.write("\n")

            file2.write(f"{i + 1}")
            file2.write(',')
            file2.write(str(results[2]))
            file2.write(',')
            file2.write(str(results[4]))
            file2.write(',')
            file2.write(str(results[3]))
            file2.write("\n")
        #print("END of Exp for N = "+str(n))
    file1.close()
    file2.close()
    print("END of Exp for N = "+str(n))


# Convert the timestamp to a datetime object
datetime_object = datetime.fromtimestamp(time.time())

# Format the datetime object as a string
formatted_datetime = datetime_object.strftime('%Y-%m-%d %H:%M:%S')

print("end: ",formatted_datetime)

