import numpy as np
#from scipy.optimize import line_search

f_calls = 0
grad_f_calls = 0 



def returns(w):
    w=np.array(w)
    
    R_bar = [0.29283428631380165, 0.38721166724539907, 0.7666054369330031, 0.6261117148659746]
    R_bar=np.array(R_bar) 
    
    return np.dot(w,R_bar)

def risk(w):
    w=np.array(w)
    cov =[[ 0.07493161,  0.01645338, -0.00706299, -0.0259072 ],
       [ 0.01645338,  0.38679314,  0.11740771,  0.1254754 ],
       [-0.00706299,  0.11740771,  0.21395927,  0.13668301],
       [-0.0259072 ,  0.1254754 ,  0.13668301,  0.2187645 ]]
    
    return w @ cov @ w

def F_x(x):
    
    global f_calls
    f_calls+=1
    
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
    
def partial_der_F_x(x,i,R_bar,cov,l,sumx):
    
    sumx_R_bar = 0
    diagSum = 0
    upperDiagSum = 0
    
    #considering the partial derivative of F_x having two components : 
    #A' = (R_bar[i]*sumx - sumx_R_bar)/((sumx)**2)
    #B' = diagSum + 2*upperDiagSum
    
    if sumx<=10**(-10):
        sumx = 10**(-10)  
    
    for j in range(0,4):
        
        sumx_R_bar = sumx_R_bar + x[j]*R_bar[j]
        
        if j == i:
            diagSum = diagSum + (2*x[j]*(sumx-x[j]))*cov[j][j]
        else:
            diagSum = diagSum + (-2)*x[j]*x[j]*cov[j][j]
            
        for k in range(j+1,4):
            
            if j == i:
                upperDiagSum = upperDiagSum + (x[k]*sumx - 2*x[j]*x[k])*cov[j][k]
            else:
                upperDiagSum = upperDiagSum + (-2)*x[j]*x[k]*cov[j][k]
    
    A_der = (R_bar[i]*sumx - sumx_R_bar)/((sumx)**2)
    B_der = (diagSum + 2*upperDiagSum) /((sumx)**3)       

    part_df_x = -(A_der - l*B_der)
    
    return part_df_x

def grad_F_x(x):
    
    global grad_f_calls
    
    grad_f_calls += 1
    
    x=np.array(x)
    
    R_bar = [0.29283428631380165, 0.38721166724539907, 0.7666054369330031, 0.6261117148659746]
    R_bar=np.array(R_bar)         

    cov =[[ 0.07493161,  0.01645338, -0.00706299, -0.0259072 ],
       [ 0.01645338,  0.38679314,  0.11740771,  0.1254754 ],
       [-0.00706299,  0.11740771,  0.21395927,  0.13668301],
       [-0.0259072 ,  0.1254754 ,  0.13668301,  0.2187645 ]]
    
    l=1.5
    sumx=np.sum(x)
    
    dfx1 = partial_der_F_x(x,0,R_bar,cov,l,sumx)
    dfx2 = partial_der_F_x(x,1,R_bar,cov,l,sumx)
    dfx3 = partial_der_F_x(x,2,R_bar,cov,l,sumx)
    dfx4 = partial_der_F_x(x,3,R_bar,cov,l,sumx)
    
    return np.array([dfx1,dfx2,dfx3,dfx4])




def bisection_interpolation(a,b):
    return ((a+b)/2)

def lineZ(xk,xk_new,t):
    return float(1-t)*np.array(xk) + float(t)*np.array(xk_new)

def bisection_line_clamp(xk,xk_new,x_lo,x_hi,epsilon):
    #bisection
    ak = x_lo
    bk = x_hi
        
    while(True):
        t = 0.5*(ak+bk)
        z=lineZ(xk,xk_new,t)
        if np.any(z<=x_lo) or np.any(z>=x_hi):
            bk = t
        else:
            ak = t
            
        if (bk-ak) < epsilon:
            break
    t = 0.5*(ak+bk)
    return lineZ(xk,xk_new,t)

def on_bounds_clamp(xk_new,x_lo,x_hi,e):
    clampedVec = np.zeros(len(xk_new))
    
    for i in range(0,len(xk_new)):
        #if around zero or out of bounds on the negative side -> clamp to 0
        around_lo_xi = ((xk_new[i]<= x_lo+e)and(xk_new[i]>=x_lo-e))
        #elif arounnd one or out of bounds bc it is > 1 -> clamp to 1
        around_hi_xi = ((xk_new[i]<= x_hi+e)and(xk_new[i]>= x_hi - e))
        #else ->  leave it as is
        if around_lo_xi or xk_new[i]< x_lo :
            clampedVec[i]= x_lo
        elif around_hi_xi or xk_new[i]> x_hi:
            clampedVec[i] = x_hi
        else:
            clampedVec[i]=xk_new[i]
    return clampedVec
    
def clamp_x(xk,xk_new,x_lo,x_hi,e):
    
    
   
    around_lo = np.any((xk_new<= x_lo + e)&(xk_new>=x_lo - e)) # range [-e,e] around x_lo=0
    around_hi = np.any((xk_new<= x_hi +e)&(xk_new>= x_hi - e))# range [1-e,1+e] around x_hi=1
    
    outOfBound_lo = np.any((xk_new<= x_lo))
    outOfBound_hi = np.any((xk_new>= x_hi))
    
    #if any ON boundary -> clamp on bounds
    if (around_lo) or (around_hi):
        #print("Clamp On Bounds")
        return on_bounds_clamp(xk_new,x_lo,x_hi,e)
   
    elif outOfBound_lo or outOfBound_hi:
       #if any out of bounds -> bisection to find acceptable vector 
       #print("Bisection Clamp")
       z = bisection_line_clamp(xk,xk_new,x_lo,x_hi,e)
       for i in range(0,len(z)):
           if z[i]<x_lo: z[i] = x_lo
           if z[i]>x_hi : z[i] = x_hi
       return z
        
    else:
        # all within bounds , return as is
        #print("Return As Is -> ALL OK")
        return xk_new   
    



def zoom(obj_func,obj_grad,xk,pk,phi_0,df_phi_0,a_lo,a_hi,c1,c2):
   
    while(True): 
        a_j = bisection_interpolation(a_lo, a_hi)
    
        phi_a_j = obj_func(xk + a_j*pk)
        phi_lo = obj_func(xk + a_lo*pk)
       
        #print("a_lo = "+str(a_lo)+" a_j = "+str(a_j)+" a_hi = "+str(a_hi))
        
        #SAFEGUARD
        #if Armijo holds and the solution is too close to the bounds, return a_j
        if (phi_a_j<= phi_0 + c1*a_j*np.dot(df_phi_0,pk)):
            if(np.abs(a_j-a_lo)<=10**(-4))and(np.abs(a_j-a_hi)<=10**(-4)):
                return a_j; 
        
        
        #if np.all(phi_a_j> phi_0 + c1*a_j*df_phi_0) or (phi_a_j>=phi_lo):
        if (phi_a_j> phi_0 + c1*a_j*np.dot(df_phi_0,pk)) or (phi_a_j>=phi_lo):
            

            a_hi = a_j
        else:
            
            
            df_phi_a_j = np.array(obj_grad(xk + a_j*pk))
            
            
            #if np.dot(df_phi_a_j,pk)>= c2*np.dot(df_phi_0,pk):
            if np.abs(np.dot(df_phi_a_j,pk))<= -c2*np.dot(df_phi_0,pk):
                return a_j
            
            if np.all(df_phi_a_j*(a_hi - a_lo)>=0):
                a_hi = a_lo
            
            a_lo = a_j
            

            
            

def lineSearch(obj_func,obj_grad,xk,pk,c1,c2,a_max):
    
    a0 = 0 
    a_prev = 0 #initialized, won't be used in the 1st iteration
    i = 0
    conseq = 0
    a_i = bisection_interpolation(a0, a_max)
    
    phi_a_prev = obj_func(xk+a_prev*pk)
    phi_0 = obj_func(xk)  # input: xk+0*pk
    df_phi_0 = np.array(obj_grad(xk)) #input: xk+0*pk
    while(True):
        
        phi_a_i = obj_func(xk+a_i*pk)
        #phi_a_prev = obj_func(xk+a_prev*pk)
     
        
        if(np.abs(phi_a_i-phi_a_prev)<=10**(-4)):
            conseq = conseq + 1 
        else:
            conseq = 0     #reset
        
        if conseq >= 10:
            
            if (phi_a_i<= phi_0 + c1*a_i*np.dot(df_phi_0,pk)):
            
                return a_i
        
        
        
        
        #if np.all(phi_a_i>phi_0 + c1*a_i*df_phi_0) or ((phi_a_i>=phi_a_prev)and(i>1)):
        if (phi_a_i> phi_0 + c1*a_i*np.dot(df_phi_0,pk)) or ((phi_a_i>=phi_a_prev)and(i>1)):
            #print("ZOOM a_prev a_i")
            return zoom(obj_func,obj_grad,xk,pk,phi_0,df_phi_0,a_prev,a_i,c1,c2)
            
        df_phi_a_i = np.array(obj_grad(xk+a_i*pk))
        #gradEvals = gradEvals + 1
            

        #if np.dot(df_phi_a_i,pk)>= c2*np.dot(df_phi_0,pk):
        if np.abs(np.dot(df_phi_a_i,pk))<= -c2*np.dot(df_phi_0,pk):
            return a_i
            
        if np.all(df_phi_a_i>=0):
            #print("ZOOM a_i a_prev")
            return zoom(obj_func,obj_grad,xk,pk,phi_0,df_phi_0,a_i,a_prev,c1,c2)
        phi_a_prev = phi_a_i
        a_prev  = a_i
        a_i = bisection_interpolation(a_i, a_max)

def posDef_check_pos_eigenval(B):
    
    #positive eigenvalues criterion
    w,v=np.linalg.eig(B)
    #print(w)
    if np.all(w>0):
        return 1
    else:
        return 0
    
    return 


def fixHessian(H,beta,k_max):
    n=np.size(H[0])
    diag_min = np.min(np.diag(H))
    if(diag_min>0):
        tau_k = beta
    else:
        tau_k = -diag_min+beta

    for k in range(0,k_max):
        B = H+tau_k*np.eye(n)
        #if(posDef_check_diag_dom(B)==1) or (posDef_check_pos_eigenval(B)== 1):
        #_,checkVal = cholesky_decomposition_check(B)
        #if checkVal == 1:
        if posDef_check_pos_eigenval(B)==0:
            prev_tau_k = tau_k
            tau_k = max(2*prev_tau_k,beta)  
        else:
            return B

def BFGS_UPDATE_H(Hk,sk,yk):
    
    n= np.size(yk)
    I = np.eye(n)
    
    #skip update and use previous Hk -> NOT THE BEST SOLUTION 
    if (np.dot(yk,sk)<=10**(-20)):
        return Hk

    rho_k = 1 / (np.dot(yk,sk))

    A = I - rho_k*(np.outer(sk,yk))
    B = I - rho_k*(np.outer(yk,sk))
    C = rho_k*(np.outer(sk,sk))

    return A@Hk@B+C
    
    


def LS_BFGS(obj_func,obj_grad,x,epsilon,c1,c2,a_max):
    
    n=len(x)    
    Hk = np.eye(n) #init inv hessian approx H0

    k=0
    invHessianFixes=0

    startPoint = x
    #restartPoints =[]
    xk=x
  
    
    printData = 0
    #tooClose = 0
    

    global f_calls

    while(True):
        

        if printData == 1:
            print("---------------------------------------------------- k = "+str(k))
        
        #experiment to escape possible local minimum
        #if tooClose == 20 :    
        #    xk = xk*0.01
        #    restartPoints.append(xk)
        
        
        if k==0:
            dfk = np.array(obj_grad(xk))
            

        
        if(np.linalg.norm(dfk)<epsilon):
            print(" ")
            print("TERMINATION CRITERION: l2 NORM of Gradient <= epsilon")
            break

        pk = (-1)*np.dot(Hk,dfk)
        #pk = -dfk
        
        #check if descending direction, if not: break, don't go into next iteration
        if(np.dot(dfk,pk)>=0)and k>=1:
            print(" ")
            print("pk NOT DESCENDING...TERMINATING") # bc if desc dir : np.dot(dfk,pk)<0!
            break
        
        
        
        
        ak = lineSearch(obj_func, obj_grad, xk, pk, c1, c2, a_max)
        #ak,_,_,_,_,_=line_search(obj_func, obj_grad,xk,pk)
        if printData == 1 :
            print("ak = "+str(ak))

        xk_new = xk + ak*pk
        xk_new = clamp_x(xk,xk_new,0,1,10**(-10))
        
        sk = xk_new - xk
        
        dfk_new = np.array(obj_grad(xk_new))
        
        
        yk = dfk_new - dfk 
        
        Hk_new = BFGS_UPDATE_H(Hk,sk,yk)
        
        if printData ==1: 
            print("xk = "+str(xk))
            print("fk = "+str(obj_func(xk)))
            
            f_calls-=1
            print("dfk = "+str(dfk))
            print("ak = "+str(ak))
            print("pk = "+str(pk))
        
        #if not positive definite - FIX IT
        #if(posDef_check_diag_dom(Hk_new)==1)or(posDef_check_pos_eigenval(Hk_new)== 1):
        #_,checkVal = cholesky_decomposition_check(Hk_new)
        #if checkVal == 1:
        if posDef_check_pos_eigenval(Hk_new)==1:
            pass
            #print("Pos Def OK")
        else:
            if printData == 1:
                print("FIX INV HESSIAN APPROX")
            Hk_new = fixHessian(Hk_new,1,20)
            invHessianFixes +=1
        
        
        #   global f_calls
        
        if f_calls > 2000:
            print("TERMINATION CRITERION: COMPUTATIONAL BUDGET RAN OUT")
            break
        
        #if np.all(xk_new - xk <=epsilon):
        #    tooClose +=  1 
            
        #    if tooClose == 50 :  
        #        print(" ")
        #        print("50 REPEATS")
        #        break
        #else:
        #    tooClose = 0
        
        
        
        xk = xk_new
        dfk = dfk_new
        Hk = Hk_new
        
      
        k += 1
    print(" ")
    print("Wolfe Line Search BFGS FINAL RESULTS --- --- --- --- --- --- --- ---")
    print(" ")
    print("INPUTS:")
    print("Starting Point  = "+str(startPoint))
    #print("Restart Points = "+str(restartPoints))
    print("Wolfe Cond. Line Search Params: "+"bounds epsilon = "+str(10**(-4))+" c1 = "+str(c1)+" c2 = "+str(c2)+" a_max = "+str(a_max))
    print("epsilon="+str(epsilon))
    print("clamp X e = "+str(10**(-10)))
    print(" ")
    print("RESULTS:")

    print("xk best = "+str(xk))
    print("fk best = "+str(obj_func(xk)))
    f_calls-=1 #do not count the print call
    #print("dfk best = "+str(dfk))
    #print("ak = "+str(ak))
    #print("pk = "+str(pk))
    w = xk/np.sum(xk)
    print("w = "+str(w))
    print("Performance(w) = "+str(returns(w)))
    print("Risk(w) = "+str(risk(w)))
    print("λ*Risk(w) = "+str(1.5*risk(w)))
    print(" ")
    print("ITERATIONS - FUNCTION and GRADIENT CALLS - H and B FIXES:")
    print("AVAILABLE COMPUTATIONAL BUDGET :  2000 (plus a few extra because it stops when f_calls > 2000)")
    print("k = "+str(k))
    print("Function Calls = "+str(f_calls))
    print("Gradient Calls = "+str(grad_f_calls))
    print("inv H fixes = "+str(invHessianFixes))

# EXPERIMENTS --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- 



p1 = [0.90362412, 0.96392195, 0.50211596, 0.69145483]
p2 = [0.70834141, 0.05891796, 0.80561353, 0.64116189]
p3 = [0.10062929, 0.56744996, 0.96195682, 0.03156802]
p4 = [0.7172232,  0.10910059, 0.28863615, 0.96416363]
p5 = [0.42583021, 0.95432923, 0.07941098, 0.28174186]
p6 = [0.60384535, 0.11496978, 0.90015125, 0.91582732]
p7 = [0.61422968, 0.16807904, 0.71007527, 0.66524447]
p8 = [0.57583732, 0.14106864, 0.53758021, 0.53470457]
p9 = [0.72509836, 0.19215341, 0.54516135, 0.02049204]
p10 = [0.60436998, 0.66122049, 0.0072185 , 0.73981168]



point = 0


if(point == 1):
    print("Point p1 => "+ str(p1))
    print("")
    LS_BFGS(F_x,grad_F_x,p1,10**(-4),10**(-4),0.9,1)
    print("*****************************************************")
if(point == 2):
    print("")
    print("Point p2 => "+ str(p2))
    print("")
    LS_BFGS(F_x,grad_F_x,p2,10**(-4),10**(-4),0.9,1)
    print("*****************************************************")
if(point == 3):
    print("")
    print("Point p3 => "+ str(p3))
    print("")
    LS_BFGS(F_x,grad_F_x,p3,10**(-4),10**(-4),0.9,1)
    print("*****************************************************")
if(point == 4):    
    print("")
    print("Point p4 => "+ str(p4))
    print("")
    LS_BFGS(F_x,grad_F_x,p4,10**(-4),10**(-4),0.9,1)
    print("*****************************************************")
if(point == 5):
    print("")
    print("Point p5 => "+ str(p5))
    print("")
    LS_BFGS(F_x,grad_F_x,p5,10**(-4),10**(-4),0.9,1)
    print("*****************************************************")
if(point == 6):
    print("")
    print("Point p6 => "+ str(p6))
    print("")
    LS_BFGS(F_x,grad_F_x,p6,10**(-4),10**(-4),0.9,1)
    print("*****************************************************")
if(point == 7):
    print("")
    print("Point p7 => "+ str(p7))
    print("")
    LS_BFGS(F_x,grad_F_x,p7,10**(-4),10**(-4),0.9,1)
    print("*****************************************************")
if(point == 8):
    print("")
    print("Point p8 => "+ str(p8))
    print("")
    LS_BFGS(F_x,grad_F_x,p8,10**(-4),10**(-4),0.9,1)
    print("*****************************************************")
if(point == 9):
    print("")
    print("Point p9 => "+ str(p9))
    print("")
    LS_BFGS(F_x,grad_F_x,p9,10**(-4),10**(-4),0.9,1)
    print("*****************************************************")
if(point == 10):    
    print("")
    print("Point p10 => "+ str(p10))
    print("")
    LS_BFGS(F_x,grad_F_x,p10,10**(-4),10**(-4),0.9,1)
    print("*****************************************************")





# --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---







# TESTING --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---      


#num_samples = 1

#bounds = [(0, 1), (0, 1), (0, 1), (0, 1)]  

#bounds = [(0.27, 0.37), (0, 0), (0.37, 0.45), (0.19, 0.35)] # w  

#bounds = [(0.58, 0.63), (0, 0), (0.74, 0.82), (0.31, 0.75)]  # x 

#random_samples = np.random.uniform(low=bounds[0][0], high=bounds[0][1], size=(num_samples, 4))    
 
#print(returns(random_samples[0])) # for w only
#print(risk(random_samples[0]))  # for w only 
#LS_BFGS(F_x,grad_F_x,random_samples[0],10**(-4),10**(-4),0.9,1) # for x 
#print("Start = "+str(random_samples[0])) 

# --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
