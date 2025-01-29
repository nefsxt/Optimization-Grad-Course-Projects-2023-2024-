import numpy as np
import random
import sys
import os


#Necessary to ensure interpreter sees the current directory 

# Get the current directory of the script
current_dir = os.path.dirname(os.path.realpath(__file__))

# Add the directory to sys.path
sys.path.append(current_dir)


import problem_functions

def rosenbrock(x):
    return (1 + x[0])**2 + 100*(x[1] - x[0]**2)**2

def tournament_selection(evals,N,N_tour,N_s):
    tour_best = []
    for i in range(0,N_s):
        tour_collect_val = []
        tour_collect_idx = []
        for j in range(0,N_tour):
            rand_idx = random.randint(0,N-1)
            tour_collect_val.append(evals[rand_idx])
            tour_collect_idx.append(rand_idx)
        best_idx = tour_collect_idx[np.argmin(np.array(tour_collect_val))]
        tour_best.append(best_idx)

    return tour_best

def linear_ranking_phi(N,s,pos):
    return (2-s) + 2*(s-1)*((pos-1)/(N-1)) 
    
def roulette_wheel_selection(evals,s,N_s,N):
    
    #f_worst_idx = np.argmax(evals)
    #f_worst_val = evals[f_worst_idx]
    
    sorted_idx = sorted(range(0,N), key=lambda x: evals[x])
    idx_of_sorted= [sorted_idx.index(i) for i in range(0,N)]
    pos_in_desc = [N-x for x in idx_of_sorted]
    
    fitness_vals = [linear_ranking_phi(N,s,pos) for pos in pos_in_desc]
    phi_hat = np.sum(np.array(fitness_vals))
    ps = [x/phi_hat for x in fitness_vals]

    
    roulette_wheel_selected=[]
    for i in range(0,N_s):
        
        R = np.random.uniform(0,1)
        rw_sum = 0
        j=-1
        while(rw_sum<=1):
            j+=1
            rw_sum+=ps[j]
            if(R<=rw_sum):
                roulette_wheel_selected.append(j)
                break
    #print("here")
    #print(roulette_wheel_selected)
    return roulette_wheel_selected


def real_val_crossover(N,cand_len,pop,selected,delta,c_prob):
    
    C = []
    #select parents
    parents = []
    u_lo=-delta
    u_hi = 1+delta
    for i in range(0,len(selected)):
        R = np.random.uniform(0,1)
        if R<=c_prob:
            
            parents.append(pop[selected[i]])
        else:
            C.append(pop[selected[i]]) 
    #if number of parents is odd -> add another random parent to make pairs feesible
    if len(parents)%2 !=0:
        rand_idx = random.randint(0,N-1)
        parents.append(pop[rand_idx])
         
    #produce offspring
    p1=0 
    while(p1!=len(parents)):
        p2 = p1+1
        parent1 = parents[p1]
        parent2 = parents[p2]
        offspring1 = []
        offspring2 = []
        #produce offspring 1 and 2
        for j in range(0,cand_len):
            rj = np.random.uniform(u_lo,u_hi)
            o1_j = rj*parent1[j] + (1-rj)*parent2[j]
            o2_j = rj*parent2[j] + (1-rj)*parent1[j]
            offspring1.append(o1_j)
            offspring2.append(o2_j)
        C.append(np.array(offspring1))
        C.append(np.array(offspring2))
        p1+=2 #move to next parent pair
    return C


def real_val_uniform_mutation(cand_len,C,m_prob):
    M = []
    for c in range(0,len(C)):
        cand = []
        for j in range(0,cand_len):
            R = np.random.uniform(0,1)
            if R<=m_prob:
                pj = C[c][j]
                alpha = 1-pj
                zj = np.random.uniform(-alpha,alpha) 
                cand_j = pj + zj # technically should result in values within search space
                
                #just in case 
                if cand_j<0.0:
                    cand_j = 0.0
                if cand_j > 1.0:
                    cand_j = 1.0
                    
                cand.append(cand_j)
            else:
                cand_j = C[c][j]
                if cand_j<0.0:
                    cand_j = 0.0
                if cand_j > 1.0:
                    cand_j = 1.0
                cand.append(cand_j)
        #print(cand) 
        M.append(np.array(cand))    
    return M

def ga_roulette_main(X,N,cand_len,s,N_s,c_prob,m_prob,delta):

    #load external data
    filepath = r'C:\Users\Nefeli\Desktop\opti_project2\customer_coordinates.xlsx'
    customerDf = problem_functions.read_client_data(filepath)
    
    #get lake coordinates
    lake_coords = problem_functions.lake_coord_to_real()


    toPrint = 0

    #Initialize population
    pop = [np.random.uniform(X[0], X[1], cand_len) for i in range(N)]
    
    #Inital Evaluation of population
    evals=[]
    for p in pop:
        decod = problem_functions.decoder(lake_coords,p)
        #print(decod)
        evals.append(problem_functions.cost_function(customerDf,decod[0],decod[1],decod[2],decod[3]))
    
    
    #evals = [rosenbrock(x) for x in pop]
    #Initalize best index and value variables
    prev_best_idx = np.argmin(evals)
    overall_best_idx = prev_best_idx

    prev_best_val = evals[prev_best_idx]
    overall_best = prev_best_val

    overall_best_cand = pop[overall_best_idx]

    #Initialize Function Evaluation Counter and last hit
    #func_evals = N
    lastHit = prev_best_idx + 1 #plus 1 to account for indexing starting from 0 


    #Debug prints
    if toPrint == 1:
        print("INITIALIZATION:")
        print("Population:")
        print(pop)
        print("Evaluations:")
        print(evals)
        print("Best Idx: "+str(prev_best_idx))
        print("Overall Best Idx: "+str(overall_best_idx))
        print("Best Value: "+str(prev_best_val))
        print("Overall Best Value: "+str(overall_best))
        print("Overall Best Candidate: "+str(overall_best_cand))
        print("*** *** *** *** *** ***")
    gen = 1
    

    
    while(gen*N<=200000):
        #print("GEN "+str(gen)+"------------------------------------")
        #print(evals)
        

        S = roulette_wheel_selection(evals,s,N_s,N)
        C = real_val_crossover(N,cand_len,pop,S,delta,c_prob)
        M = real_val_uniform_mutation(cand_len,C,m_prob)
        #print(S)
        #print(C)
        #print(M)
        #evaluation
        m_evals=[]
        for i in range(0,len(M)):
            #m_eval_i = rosenbrock(M[i])
            #print(M[i])
            decod = problem_functions.decoder(lake_coords,M[i])
            m_eval_i = problem_functions.cost_function(customerDf,decod[0],decod[1],decod[2],decod[3])
        
            if m_eval_i < overall_best:
                #print("OVERALL REPLACED")
                overall_best = m_eval_i
                overall_best_idx = i
                overall_best_cand = M[i]
                lastHit = (gen - 1)*N + (i+1) # plus one to account for 0 start index
            m_evals.append(m_eval_i)

        #update population    
        evals[:] = m_evals
        pop[:] = M 
   
        gen_best_idx = np.argmin(evals)
        gen_best =evals[gen_best_idx]
   

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
          

     
    return [overall_best_cand,decode_best[4],overall_best,lastHit,feasibility]    
        

def ga_tournament_main(X,N,cand_len,N_tour,N_s,c_prob,m_prob,delta):

    #load external data
    filepath = r'C:\Users\Nefeli\Desktop\opti_project2\customer_coordinates.xlsx'
    customerDf = problem_functions.read_client_data(filepath)
    
    #get lake coordinates
    lake_coords = problem_functions.lake_coord_to_real()


    toPrint = 0

    #Initialize population
    pop = [np.random.uniform(X[0], X[1], cand_len) for i in range(N)]
    
    #Inital Evaluation of population
    evals=[]
    for p in pop:
        decod = problem_functions.decoder(lake_coords,p)
        #print(decod)
        evals.append(problem_functions.cost_function(customerDf,decod[0],decod[1],decod[2],decod[3]))
    
    
    #evals = [rosenbrock(x) for x in pop]
    #Initalize best index and value variables
    prev_best_idx = np.argmin(evals)
    overall_best_idx = prev_best_idx

    prev_best_val = evals[prev_best_idx]
    overall_best = prev_best_val

    overall_best_cand = pop[overall_best_idx]

    #Initialize Function Evaluation Counter and last hit
    #func_evals = N
    lastHit = prev_best_idx + 1 #plus 1 to account for indexing starting from 0 


    #Debug prints
    if toPrint == 1:
        print("INITIALIZATION:")
        print("Population:")
        print(pop)
        print("Evaluations:")
        print(evals)
        print("Best Idx: "+str(prev_best_idx))
        print("Overall Best Idx: "+str(overall_best_idx))
        print("Best Value: "+str(prev_best_val))
        print("Overall Best Value: "+str(overall_best))
        print("Overall Best Candidate: "+str(overall_best_cand))
        print("*** *** *** *** *** ***")
    gen = 1
    

    
    while(gen*N<=200000):
        #print("GEN "+str(gen)+"------------------------------------")
        #print(evals)
        
        S =  tournament_selection(evals,N,N_tour,N_s)
        C = real_val_crossover(N,cand_len,pop,S,delta,c_prob)
        M = real_val_uniform_mutation(cand_len,C,m_prob)
        #print(S)
        #print(C)
        #print(M)
        #evaluation
        m_evals=[]
        for i in range(0,len(M)):
            #m_eval_i = rosenbrock(M[i])
            #print(M[i])
            decod = problem_functions.decoder(lake_coords,M[i])
            m_eval_i = problem_functions.cost_function(customerDf,decod[0],decod[1],decod[2],decod[3])
        
            if m_eval_i < overall_best:
                #print("OVERALL REPLACED")
                overall_best = m_eval_i
                overall_best_idx = i
                overall_best_cand = M[i]
                lastHit = (gen - 1)*N + (i+1) # plus one to account for 0 start index
            m_evals.append(m_eval_i)

        #update population    
        evals[:] = m_evals
        pop[:] = M 
   
        gen_best_idx = np.argmin(evals)
        gen_best =evals[gen_best_idx]
   
      
        if toPrint == 1:
            if gen%100 == 0:
                print("Gen ",gen) 
   
   
        prev_best_val = gen_best
        prev_best_idx = gen_best_idx  
   
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
            

     
    return [overall_best_cand,decode_best[4],overall_best,lastHit,feasibility]   


def test_tour_ga():
    # For rosenbrock
    N=200
    cand_len = 57 
    N_s = 200
    N_tour = 25
    c_prob = 0.5
    m_prob = 0.2
    delta = 0.25
    X = [0.0,1.0]
    
     
    return ga_tournament_main(X,N,cand_len,N_tour,N_s,c_prob,m_prob,delta)

def test_rw_ga():
    # For rosenbrock
    N=200
    cand_len = 57 
    s=2
    N_s = 200
    c_prob = 0.5
    m_prob = 0.2
    delta = 0.25
    X = [0.0,1.0]
     
    return ga_roulette_main(X,N,cand_len,s,N_s,c_prob,m_prob,delta)
        

#test_tour_ga()
#test_rw_ga()
