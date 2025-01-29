import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

def F_x(x):
    

    
    x=np.array(x)
    
    R_bar = [0.29283428631380165, 0.38721166724539907, 0.7666054369330031, 0.6261117148659746]
    R_bar=np.array(R_bar)         

    cov =[[ 0.07493161,  0.01645338, -0.00706299, -0.0259072 ],
       [ 0.01645338,  0.38679314,  0.11740771,  0.1254754 ],
       [-0.00706299,  0.11740771,  0.21395927,  0.13668301],
       [-0.0259072 ,  0.1254754 ,  0.13668301,  0.2187645 ]]
    
    
    sumx=np.sum(x)
    #consider the form F_x = -(A-lamba*B)
    A = 0
    B = 0
    l = 1.5 #lambda
    
    #calculate A and B - > done in the same loop for economy 
    for j in range(0,4):
        
        A = A + x[j]*R_bar[j]
        
        for k in range(0,4):
            B = B + x[j]*x[k]*cov[j][k]
     
    if sumx<=10**(-10):
        sumx = 10**(-10)        
    A = A/sumx
    B = B/(sumx**2)
    f_x = -(A -l*B)       
    return f_x




num_samples = 1000
bounds = [(0, 1), (0, 1), (0, 1), (0, 1)] 

median_f_vals = np.empty((1000,))

random_samples = np.random.uniform(low=bounds[0][0], high=bounds[0][1], size=(num_samples, 4))

for i in range(10):
    function_values = np.apply_along_axis(F_x, 1, random_samples)
    median_f_vals= median_f_vals  + function_values
    #sampled_data = list(zip(random_samples, function_values))
median_f_vals  = median_f_vals /10


#APPLY K-MEANS TO GET 11 GROUPS OF VALUES -------------------------------------

n_clusters = 11

function_values = function_values.reshape(-1, 1)


kmeans = KMeans(n_clusters=n_clusters, random_state=42)
cluster_labels = kmeans.fit_predict(function_values)

plt.scatter(np.arange(len(function_values)), function_values, c=cluster_labels, cmap='viridis')
plt.xlabel('Sample Index')
plt.ylabel('Obj.Function Values')
plt.title('Obj. Function Median Values (K-Means, K=11)')
plt.show()
