import numpy as np

import sys
import os


# Get the current directory of the script
current_dir = os.path.dirname(os.path.realpath(__file__))

# Add the directory to sys.path
sys.path.append(current_dir)

import problem_functions


def rosenbrock(x):
    return (1 + x[0])**2 + 100*(x[1] - x[0]**2)**2



def get_NB_best(nb_r,current_idx,N,best_evals,overall_best_idx):
    
    bestVal = 0
    rightVal = 0
    leftVal=0
    bestIdx = 0
    rightIdx = 0
    leftIdx = 0
    
    # l-best if nb_r > 0 
    if nb_r > 0:
        #start with the first neighbor 
        bestVal = best_evals[current_idx % N]
        bestIdx = current_idx % N
        
        #search through neighbors on the right
        for i in range(1,nb_r+1):
            #check right neighbor 
            rightVal = best_evals[(current_idx+i)%N] 
            rightIdx = (current_idx+i)%N
            
            if rightVal < bestVal:
                bestVal = rightVal 
                bestIdx  = rightIdx
                
            #check left neighbor
            leftVal = best_evals[(current_idx-i + N)%N] 
            leftIdx = (current_idx - i + N)%N
            
            if leftVal < bestVal:
                bestVal = leftVal 
                bestIdx  = leftIdx

        return bestIdx
    else:
        return overall_best_idx
        
def pso_main(X,N,cand_len,chi,c1,c2,alpha,nb_r):
    
    #load external data
    filepath = r'C:\Users\Nefeli\Desktop\opti_project2\customer_coordinates.xlsx'
    customerDf = problem_functions.read_client_data(filepath)
    
    #get lake coordinates
    lake_coords = problem_functions.lake_coord_to_real()
    


    toPrint = 0
    
    best_pos = []
    best_evals = []
    
    
    
    v_max = alpha*(X[1]-X[0])
    
    #Initialize swarm, velocities and best positions
    swarm = [np.random.uniform(X[0], X[1], cand_len) for i in range(N)]
    veloc = [np.random.uniform(-v_max, v_max, cand_len) for i in range(N)]
    best_pos[:] = swarm
    
    #Inital Evaluation of swarm
    #evals = [rosenbrock(x) for x in swarm]
    
    #Inital Evaluation of population
    evals=[]
    for s in swarm:
        decod = problem_functions.decoder(lake_coords,s)
        #print(decod)
        evals.append(problem_functions.cost_function(customerDf,decod[0],decod[1],decod[2],decod[3]))
    
    best_evals[:] = evals
    
    
    #Initalize best index and value variables
    prev_best_idx = np.argmin(evals)
    overall_best_idx = prev_best_idx

    prev_best_val = evals[prev_best_idx]
    overall_best = prev_best_val

    overall_best_cand = best_pos[overall_best_idx]

    #last hit initialization
    lastHit = prev_best_idx + 1 #plus 1 to account for indexing starting from 0 


    #Debug prints
    if toPrint == 1:
        print("INITIALIZATION:")
        print("Swarm:")
        print(swarm)
        print("Velocities:")
        print(veloc)
        print("Best Positions:")
        print(best_pos)
        print("Evaluations:")
        print(evals)
        print("Best Idx: "+str(prev_best_idx))
        print("Overall Best Idx: "+str(overall_best_idx))
        print("Best Value: "+str(prev_best_val))
        print("Overall Best Value: "+str(overall_best))
        print("Overall Best Candidate: "+str(overall_best_cand))
        print("*** *** *** *** *** ***")
    
    gen = 1

    restart_intervals = [0.05,0.1,0.2,0.4,0.8]
    restart_counter = 0
    no_improvement = 0
    #while(gen*N<=1000):
    while(gen*N<=200000):
        #print("GEN "+str(gen)+"------------------------------------")
        #print(evals)
    
        #restart check 
        if no_improvement == (restart_intervals[restart_counter]*(200000/N)):
            
            #print("restart",restart_counter)
        
            swarm = [np.random.uniform(X[0], X[1], cand_len) for i in range(N)]
            veloc = [np.random.uniform(-v_max, v_max, cand_len) for i in range(N)]    
        
            evals=[]
            for s in swarm:
                
                decod = problem_functions.decoder(lake_coords,s)
                #print(decod)
                evals.append(problem_functions.cost_function(customerDf,decod[0],decod[1],decod[2],decod[3]))
            
                prev_best_idx = np.argmin(evals)
                prev_best_val = evals[prev_best_idx]
                
            restart_counter+=1
            
        for i in range(0,N):
            
            #print("Before "+str(swarm[i]))
            
            for j in range(0,cand_len):
                
                #velocity update
                r1 = np.random.rand()
                r2 = np.random.rand()
                
                nb_best = get_NB_best(nb_r,i,N,best_evals,overall_best_idx)
                #nb_best = overall_best_idx
                
                vel_new = chi*(veloc[i][j]+r1*c1*(best_pos[i][j]-swarm[i][j])+r2*c2*(best_pos[nb_best][j]-swarm[i][j]))
                #print(vel_new)
                #velocity clamp
                if vel_new < -v_max:
                    vel_new  = -v_max
                
                if vel_new > v_max:
                        vel_new = v_max
                
                #position update
                new_pos = swarm[i][j]+vel_new
                
                #position clamp
                if new_pos < X[0]:
                    new_pos = X[0]
                
                if new_pos > X[1]:
                    new_pos = X[1]
                    
                #print("Before "+str(swarm[i][j]))
                swarm[i][j] = new_pos
                #print("After "+str(swarm[i][j]))
                veloc[i][j] = vel_new
                
                
            #print("After "+str(swarm[i]))    
            
            #evaluate paritcle
            #evals[i]=rosenbrock(swarm[i])
            decod = problem_functions.decoder(lake_coords,swarm[i])
            #print(decod)
            evals[i]=problem_functions.cost_function(customerDf,decod[0],decod[1],decod[2],decod[3])
            
            #print(evals[i])
            #check for update best_pos
            if evals[i]<best_evals[i]:
                best_evals[i] = evals[i]
                best_pos[i]= swarm[i]
            
            #check for overalls best update 
            #if best_evals[i]< overall_best:
            #    overall_best = best_evals[i]
            #    overall_best_cand = best_pos[i]
            #    overall_best_idx = i
            #    #print(i)
            #    lastHit = (gen - 1)*N + (i+1) # plus one to account for 0 start index
        
        gen_best_idx = np.argmin(best_evals)
        gen_best =best_evals[gen_best_idx]
        
        if gen_best < overall_best:
           
            overall_best = gen_best
            overall_best_cand = best_pos[gen_best_idx]
            overall_best_idx = gen_best_idx
        #    #print(i)
            lastHit = (gen - 1)*N + (gen_best_idx+1) # plus one to account for 0 start index

        #no improvement check
        if prev_best_val == gen_best:
            no_improvement +=1
        else:
            no_improvement = 0

        prev_best_val = gen_best
        prev_best_idx = gen_best_idx

        if toPrint == 1:
            if gen%100 == 0:
                print("Gen ",gen)
            
        gen+=1
        
    decode_best = problem_functions.decoder(lake_coords,overall_best_cand)
    
    penalties = decode_best[2]
    if np.sum(penalties) == 0:
        feasibility = 1
    else:
        feasibility = 0
    
    if toPrint == 1:
        print("Overall Best Value: "+str(overall_best))
        print("Overall Best Candidate: "+str(overall_best_cand))
        print("Last Hit: "+str(lastHit))
        print("Restarts: ",restart_counter)
    
    return [overall_best_cand,decode_best[4],overall_best,lastHit,feasibility]

########## TEST BENCH FUNCTIONS ###############################################

def test_gbest():
    N=200
    cand_len = 57
    X = [0.0,1.0]
    c1 = 2.05
    c2 = 2.05
    chi = 0.729
    alpha = 0.005
    nb_r=0
    print(pso_main(X,N,cand_len,chi,c1,c2,alpha,nb_r))
        
        
def test_lbest():
    N=200
    cand_len = 57
    X = [0.0,1.0]
    c1 = 2.05
    c2 = 2.05
    chi = 0.729
    alpha = 0.005
    nb_r=2
    print(pso_main(X,N,cand_len,chi,c1,c2,alpha,nb_r))

#test_lbest()
#test_gbest()