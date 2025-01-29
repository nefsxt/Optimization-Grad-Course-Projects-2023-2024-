
import numpy as np

num_samples = 1

# Define the bounds for each dimension
bounds = [(0, 1), (0, 1), (0, 1), (0, 1)]  

    
    

for i in range(1,11):
    
    random_samples = np.random.uniform(low=bounds[0][0], high=bounds[0][1], size=(num_samples, 4))
    print("p"+str(i)+" = "+str(np.array(random_samples[0])))