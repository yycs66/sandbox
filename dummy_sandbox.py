# This is a test dummy algorithm to get the opportunity cost curves
from ortools.linear_solver import pywraplp
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def scheduler(prices):
    # battery parameters
    capacity=18
    ch_limit=20
    dis_limit=20
    effcy=0.9
    
    number_step =len(prices)
    # [START solver]
    # Create the linear solver with the GLOP backend.
    solver = pywraplp.Solver.CreateSolver("GLOP")
    if not solver:
        return
    # [END solver]
#solver = pywraplp.Solver("B", pywraplp.Solver.GLOP_LINEAR_PROGRAMMING)

#Variables: all are continous
    charge = [solver.NumVar(0.0, ch_limit, "c"+str(i)) for i in range(number_step)]
    discharge = [solver.NumVar(0, dis_limit,  "d"+str(i)) for i in range(number_step)]
    soc = [solver.NumVar(0.0, capacity, "b"+str(i)) for i in range(number_step+1)]
    soc[0]=0

#Objective function
    solver.Minimize(
        sum(prices[i]*(charge[i]-discharge[i]) for i in range(number_step)))
    for i in range(number_step):
      solver.Add(soc[i] +effcy*charge[i] -discharge[i]==soc[i+1])
  # create the constaints on soc
    solver.Solve()
    print("Solution:")
    print("The Storage's profit =", solver.Objective().Value())
    charge_list=[]
    discharge_list=[]
    for i in range(number_step):
      charge_list.append(charge[i].solution_value())
      discharge_list.append(discharge[i].solution_value())
    return charge_list,discharge_list
    #print("charge =",charge_list)
    #print("discharge=", discharge_list)

def cost_curves(charge,discharge,effcy,prices):
    #calculate the opportunity cost for charge/discharge
    #[ch,dis]=scheduler(prices)
    #effcy=0.9
    #find the index of time period for charging/discharging
    index_ch=np.asarray(np.where(np.array(charge)>0)).flatten()
    ch_index =np.where(np.array(charge)>0)[0]
    index_dis=np.asarray(np.where(np.array(discharge)>0)).flatten()
    dis_index =np.where(np.array(discharge)>0)[0]

    idx0 =np.append(dis_index,ch_index)
    idx = np.sort(idx0)

    # create two list for charging/discharging opportunity costs
    oc_dis_list=[]
    oc_ch_list=[]


    for i in range(len(prices)):
        if i< index_ch[0]:
        # for hours before the first charge
        # opportunity cost for charging
            max_ch_temp =prices[i:index_ch[0]].max()
            oc_ch =max(max_ch_temp*effcy, prices[index_ch[0]])
            oc_ch_list.append(oc_ch)
            
            #opportunity cosy for discharging
            if i==0:
                # Hour 0, the oc_dis = to oc_ch+0.01
                oc_dis = oc_ch+0.01
                oc_dis_list.append(oc_dis)
            elif i==1:  
                oc_dis =prices[0]/effcy
                oc_dis_list.append(oc_dis)
            elif i>1 :
                oc_dis= prices[0:i].min()/effcy
                oc_dis_list.append(oc_dis)
            

      
      
        # For the hours with a scheduled charge
        elif i==index_ch[0]:
            #oc_ch updates
            arr =np.delete(prices[0:index_dis[0]], index_ch[0])
            min_oc_temp = arr.min()
            oc_ch = min(min_oc_temp, effcy*prices[index_dis[0]])
            oc_ch_list.append(oc_ch)
            #oc_dis updates
            arr1 =prices[0:index_ch[0]].min()
            arr2 =prices[(index_ch[0]+1):index_dis[0]].min()
            oc_dis =(-prices[index_ch[0]]+arr1+arr2)/effcy
            oc_dis_list.append(oc_dis)
            
        
        
        elif i== index_ch[0]+1:
            oc_ch = max(prices[index_ch[0]], prices[i]*effcy)
            oc_ch_list.append(oc_ch)
            oc_dis =min(prices[index_dis[0]], prices[(i+1):index_dis[0]].min()/effcy)
            oc_dis_list.append(oc_dis)
        
       
        elif i>index_ch[0]+1 and i< index_dis[0]:
        
            max_ch_temp = prices[(index_ch[0]+1):i]
            max_ch_temp =max_ch_temp.max()
            oc_ch = max(max_ch_temp*effcy, prices[index_ch[0]])
            oc_ch_list.append(oc_ch)
            if i< index_dis[0]-2:
                oc_dis = min(prices[index_dis[0]], prices[(i+1):index_dis[0]].min()/effcy)
                oc_dis_list.append(oc_dis)
            elif i== index_dis[0]-2:
                oc_dis = min(prices[index_dis[0]], prices[i+1]/effcy)
                oc_dis_list.append(oc_dis)
            elif i== index_dis[0]-1:
                oc_dis = min(prices[index_dis[0]],prices[index_dis[0]]/effcy)
                oc_dis_list.append(oc_dis)
              
       
        elif i== index_dis[0]:
            arr1 = prices[(index_ch[0]+1):index_dis[0]].max()
            arr2 = prices[(index_dis[0]+1):24].max()
            oc_ch =(-prices[index_dis[0]] +arr1 +arr2)*effcy
            oc_ch_list.append(oc_ch)
            oc_dis = max(prices[index_ch[0]]/effcy, prices[(index_dis[0]+1):24].max())
            oc_dis_list.append(oc_dis)
        elif i>index_dis[0] and i<23:
            oc_ch =prices[(i+1):24].max()*effcy
            oc_ch_list.append(oc_ch)
            oc_dis = max(prices[index_ch[0]]/effcy, prices[(index_dis[0]+1):24].max())
            oc_dis_list.append(oc_dis)
        elif i==len(prices)-1:
            oc_ch=0
            oc_ch_list.append(oc_ch)
            oc_dis =max(prices[index_ch[0]]/effcy, prices[i])
            oc_dis_list.append(oc_dis)
            print("Cost curves for charging:", oc_ch_list)
            print("Cost curves for discharging:", oc_dis_list)
            x =np.linspace(0,23,1)
            x.shape
            plt.plot(prices,'-bo',label='Forecast Prices')
            plt.plot(oc_dis_list,'--r*',label= 'Oppurtunity Cost for Discharge')
            plt.plot(oc_ch_list,'.g',label= 'Oppurtunity Cost for charge')
            plt.legend()
            plt.xlabel('Hours')
            plt.ylabel('Prices ($/MW)')

      

       
       
       
       
       

    