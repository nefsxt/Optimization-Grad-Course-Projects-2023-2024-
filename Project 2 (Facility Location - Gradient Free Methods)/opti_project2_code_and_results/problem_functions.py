import numpy as np 
import pandas as pd

def read_client_data(filepath):
    df = pd.read_excel(filepath)  
    
    #print(df.head(10))
    #print(df.Lat.iloc[1])
    
    return df
    


def C_pkg(d):
    if d <= 500:
        return 0.05
    elif d>500 and d<=1000:
        return 0.04
    else:
        return 0.03
    
def totalDemand(customerDf,J_i):
    
    demand_sum = 0
    
    for j in J_i:
        demand_sum+= customerDf.Demand.iloc[j]
    
    return demand_sum 



def haversine(lat_i,long_i,lat_j,long_j):
    R = 6371 #Earth radius in km  -> result is also in km
    m = 0.0174532925
    
    delta_lat = (lat_j-lat_i)*m
    delta_long = (long_j - long_i)*m
    
    return 2*R*np.arcsin(np.sqrt(((np.sin(0.5*delta_lat))**2)+np.cos(lat_j*m)*np.cos(lat_i*m)*((np.sin(0.5*delta_long))**2)))
    
    
def totalApproxDistance(customerDf,J_i,lat_i,long_i):
    #find total approx. distance in km
    
    dist_sum = 0
    
    for j in J_i:
        
        lat_j = customerDf.Lat.iloc[j]
        long_j = customerDf.Lon.iloc[j]
        
        dist_sum+= haversine(lat_i,long_i,lat_j,long_j)
    
    return dist_sum 



def real_to_int_three(y):
    #warehouse index decoder and warehouse assignment index decoder 
    #converts a real value to the corresponding integer value
    #for U = M = 3
    
    bound1 =1/3 
    bound2 = 2/3
    if y >=0.0 and y<=bound1:
        return 1
    elif y>bound1 and y<=bound2:
        return 2
    elif y>bound2 and y<=1.0:
        return 3
    else:
        return np.nan

def real_to_int_two(y):
    # warehouse assignment index decoder 
    #converts a real value to the corresponding integer value
    #for U = M = 2
    
    bound1 =1/2 
    if y >=0.0 and y<=bound1:
        return 1
    elif y>bound1 and y<=1.0:
        return 2
    else:
        return np.nan


def real_to_coord(lat_real,long_real):
    lat_max = 39.7506861
    long_max = 20.9345622
    lat_min = 39.6003201    
    long_min = 20.7660648
    
    delta_lat = lat_max - lat_min
    delta_long = long_max - long_min
    
    lat_i = lat_real*delta_lat + lat_min
    long_i = long_real*delta_long + long_min
    
    return [lat_i,long_i]
    
    
def lake_coord_to_real():
    lat_max = 39.7506861
    long_max = 20.9345622
    lat_min = 39.6003201    
    long_min = 20.7660648
    
    #lake vertices in lat-long form
    sw_lat = 39.688367
    sw_long = 20.838671

    e_lat = 39.666708
    e_long = 20.929109
    
    s_lat = 39.632539
    s_long = 20.901448

    
    delta_lat = lat_max - lat_min
    delta_long = long_max - long_min
    
    sw_r_lat = (sw_lat-lat_min)/delta_lat
    sw_r_long =(sw_long - long_min)/delta_long
    
    e_r_lat = (e_lat-lat_min)/delta_lat
    e_r_long = (e_long - long_min)/delta_long
    
    s_r_lat = (s_lat-lat_min)/delta_lat
    s_r_long = (s_long - long_min)/delta_long
    
    return [[sw_r_lat,sw_r_long],[e_r_lat,e_r_long],[s_r_lat,s_r_long]]
    
    

def barycentric_coord_lake_bound_check(lake_coords,x,y):
    # tests if point p with cartesian coord. (x,y) is within the triangle that described the lake
    #the test is based on barycentric coordinates
    #returns the coressponding penalty value : 1000 if within or on the lake triangle, 0 if outside
    
    #references:
        #https://en.wikipedia.org/wiki/Barycentric_coordinate_system 
        #https://www.scratchapixel.com/lessons/3d-basic-rendering/ray-tracing-rendering-a-triangle/barycentric-coordinates.html
    
    
    #lake coord vertices in cartesian form 
    x1=lake_coords[0][0]
    y1=lake_coords[0][1]
    
    x2=lake_coords[1][0]
    y2=lake_coords[1][1]
    
    x3=lake_coords[2][0]
    y3=lake_coords[2][1]
    
    #transform the carterisian coordinates to barycentric
    denom = (y2-y3)*(x1-x3) + (x3-x2)*(y1-y3)
    b1 = ((y2-y3)*(x-x3)+(x3-x2)*(y-y3))/denom
    b2 = ((y3-y1)*(x-x3)+(x1-x3)*(y-y3))/denom
    b3 = 1-b1-b2
    
    
    if (b1>=0 and b1<=1) and (b2>=0 and b2<=1) and (b3>=0 and b3<=1) and ((b1+b2+b3)==1):
        #point in or on triangle
        return 1000
    else:
        return 0
    
 
    
def decoder(lake_coords,x):
    
    decoded_vector = [0]*57
    
    # find M int value 
    M = real_to_int_three(x[0])
    decoded_vector[0] = M
    
    #print(M)
    
    
    #init coords, penalties and client assignment lists
    coord1 = []
    coord2 = []
    coord3 = []
    
    pen1 = 0
    pen2 = 0
    pen3 = 0
    
    J1 = []
    J2 = []
    J3 = []
    
    if M == 1:
        # decode coordinate data 
        coord1 = real_to_coord(x[1], x[2]) 
        decoded_vector[1] = coord1[0]
        decoded_vector[2] = coord1[1]
        decoded_vector[3:7] = [np.nan] * len(decoded_vector[3:7])
        
        #check for feasibility using the lake check
        pen1 = barycentric_coord_lake_bound_check(lake_coords,x[1],x[2])

        #decode assignments 
        decoded_vector[7:57] = [1]*len(decoded_vector[7:57])
        
        J1 = [x for x in range(0,50)]
        
    elif M == 2:
        
        # decode coordinate data 
        coord1 = real_to_coord(x[1], x[2]) 
        decoded_vector[1] = coord1[0]
        decoded_vector[2] = coord1[1]
        
        coord2 = real_to_coord(x[3], x[4]) 
        decoded_vector[3] = coord2[0]
        decoded_vector[4] = coord2[1]
       
        decoded_vector[5:7] = [np.nan] * len(decoded_vector[5:7])
        
        #check for feasibility using the lake check
        pen1 = barycentric_coord_lake_bound_check(lake_coords,x[1],x[2])
        pen2 = barycentric_coord_lake_bound_check(lake_coords,x[3],x[4])
        
        #decode assignments
        for i in range(7,57):
            decoded_vector[i] = real_to_int_two(x[i])
            if decoded_vector[i] == 1:
                J1.append(i-7)
            elif decoded_vector[i]==2:
                J2.append(i-7)
            else:
                print("Assignment Error for case M = 2")
        
    elif M == 3:
        
        # decode coordinate data 
        coord1 = real_to_coord(x[1], x[2]) 
        decoded_vector[1] = coord1[0]
        decoded_vector[2] = coord1[1]
        
        coord2 = real_to_coord(x[3], x[4]) 
        decoded_vector[3] = coord2[0]
        decoded_vector[4] = coord2[1]
        
        coord3 = real_to_coord(x[5], x[6]) 
        decoded_vector[5] = coord3[0]
        decoded_vector[6] = coord3[1]
        
        #check for feasibility using the lake check
        pen1 = barycentric_coord_lake_bound_check(lake_coords,x[1],x[2])
        pen2 = barycentric_coord_lake_bound_check(lake_coords,x[3],x[4])
        pen3 = barycentric_coord_lake_bound_check(lake_coords,x[5],x[6])
        #decode assignments
        for i in range(7,57):
            decoded_vector[i] = real_to_int_three(x[i])
            #print("decoded vec = ", decoded_vector[i])
            if decoded_vector[i] == 1:
                J1.append(i-7)
            elif decoded_vector[i]==2:
                J2.append(i-7)
            elif decoded_vector[i]==3:
                J3.append(i-7)
            else:
                print("Assignment Error for case M = 3")
        
    else:
        print("Error in Decoder, Invalid M Value")
        
    return [M,[coord1,coord2,coord3],[pen1,pen2,pen3],[J1,J2,J3],decoded_vector]     
 
    
def cost_function(customerDf,M,coords,penalties,assignments):
    C_km = 1.97 
    C_op = 20
    
    if M == 1:
        sd1 = totalDemand(customerDf, assignments[0])
        C1 = C_op + sd1*C_pkg(sd1) +totalApproxDistance(customerDf,assignments[0], coords[0][1],coords[0][1])*C_km
        return np.sum([C1+penalties[0],0,0])
    elif M==2:
        sd1 = totalDemand(customerDf, assignments[0])
        sd2 = totalDemand(customerDf, assignments[1])
        C1 = C_op + sd1*C_pkg(sd1) +totalApproxDistance(customerDf,assignments[0], coords[0][0],coords[0][1])*C_km
        C2 = C_op + sd2*C_pkg(sd2) +totalApproxDistance(customerDf,assignments[1], coords[1][0],coords[1][1])*C_km
        return np.sum([C1+penalties[0],C2+penalties[1],0])
    elif M==3:
        sd1 = totalDemand(customerDf, assignments[0])
        sd2 = totalDemand(customerDf, assignments[1])
        sd3 = totalDemand(customerDf, assignments[2])
        C1 = C_op + sd1*C_pkg(sd1) +totalApproxDistance(customerDf,assignments[0], coords[0][0],coords[0][1])*C_km
        C2 = C_op + sd2*C_pkg(sd2) +totalApproxDistance(customerDf,assignments[1], coords[1][0],coords[1][1])*C_km
        C3 = C_op + sd3*C_pkg(sd3) +totalApproxDistance(customerDf,assignments[2], coords[2][0],coords[2][1])*C_km
        return np.sum([C1+penalties[0],C2+penalties[1],C3+penalties[2]])

    else: 
        return np.nan
    
    
def test_bench():    
    
    filepath = r'C:\Users\Nefeli\Desktop\opti_project2\customer_coordinates.xlsx'
    customerDf = read_client_data(filepath)
    print("Total Demand Test: ,",totalDemand(customerDf, [1,5,7,9,36]))


    lat_i = 53.32055555555556
    long_i = -1.7297222222222221

    lat_j= 53.31861111111111
    long_j = -1.6997222222222223

    print("Haversine test: ",haversine(lat_i,long_i,lat_j,long_j))
    print("total approx dist test: ", totalApproxDistance(customerDf,[1,5,7,9,36],lat_i,long_i))
    print("Real to three test: ",real_to_int_three(0.097))
    print("Real to lat long: ",real_to_coord(0.823, 0.695))

    lake_coords = lake_coord_to_real()
    x = 0.585550589893984
    y = 0.4309039783403356
    print("Lake Coordinates (Real): ",lake_coords)
    print("Barycentric coordinates check: ",barycentric_coord_lake_bound_check(lake_coords,x,y))


    x_test = np.array([0.097, 0.823, 0.695, 0.317, 0.950, 0.034, 0.439, 0.382, 0.766, 0.795, 0.187, 0.490, 0.446, 0.646, 0.709])
    print("decoder res: ",decoder(lake_coords,x_test))
    decod = decoder(lake_coords,x_test)

    print("cost test value for x of len = 15 ",cost_function(customerDf,decod[0],decod[1],decod[2],decod[3]))
    
    
    
#test_bench()    
