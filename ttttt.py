#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 26 15:18:28 2022

@author: wenqing
"""

# Import 

import numpy as np
import matplotlib.pyplot as plt
from degree_freedom_queen import *
from degree_freedom_king1 import *
from degree_freedom_king2 import *
from generate_game import *
from Chess_env import *


size_board = 4
#%%
## INITIALISE THE ENVIRONMENT

env=Chess_Env(size_board)
#%%
## PRINT 5 STEPS OF AN EPISODE CONSIDERING A RANDOM AGENT

S,X,allowed_a=env.Initialise_game()                       # INTIALISE GAME

print(S)                                                  # PRINT CHESS BOARD (SEE THE DESCRIPTION ABOVE)

print('check? ',env.check)                                # PRINT VARIABLE THAT TELLS IF ENEMY KING IS IN CHECK (1) OR NOT (0)
print('dofk2 ',np.sum(env.dfk2_constrain).astype(int))    # PRINT THE NUMBER OF LOCATIONS THAT THE ENEMY KING CAN MOVE TO


for i in range(5):
    
    a,_=np.where(allowed_a==1)                  # FIND WHAT THE ALLOWED ACTIONS ARE
    a_agent=np.random.permutation(a)[0]         # MAKE A RANDOM ACTION
    print("action",a_agent)
    S,X,allowed_a,R,Done=env.OneStep(a_agent)   # UPDATE THE ENVIRONMENT
    print("allowed_a",np.shape(allowed_a),allowed_a)
    
    ## PRINT CHESS BOARD AND VARIABLES
    print('')
    print("S",S)
    print(R,'', Done)
    print('check? ',env.check)
    print('dofk2 ',np.sum(env.dfk2_constrain).astype(int))
    
    
    # TERMINATE THE EPISODE IF Done=True (DRAW OR CHECKMATE)
    if Done:
        break


#%%
# PERFORM N_episodes=1000 EPISODES MAKING RANDOM ACTIONS AND COMPUTE THE AVERAGE REWARD AND NUMBER OF MOVES 

S,X,allowed_a=env.Initialise_game()
N_episodes=1000

# VARIABLES WHERE TO SAVE THE FINAL REWARD IN AN EPISODE AND THE NUMBER OF MOVES 
R_save_random = np.zeros([N_episodes, 1])
N_moves_save_random = np.zeros([N_episodes, 1])

for n in range(N_episodes):
    
    S,X,allowed_a=env.Initialise_game()     # INITIALISE GAME
    Done=0                                  # SET Done=0 AT THE BEGINNING
    i=1                                     # COUNTER FOR THE NUMBER OF ACTIONS (MOVES) IN AN EPISODE
    
    # UNTIL THE EPISODE IS NOT OVER...(Done=0)
    while Done==0:
        
        # SAME AS THE CELL BEFORE, BUT SAVING THE RESULTS WHEN THE EPISODE TERMINATES 
        
        a,_=np.where(allowed_a==1)
        a_agent=np.random.permutation(a)[0]

        S,X,allowed_a,R,Done=env.OneStep(a_agent)
        print("R",R)
        
        
        if Done:
            
            R_save_random[n]=np.copy(R)
            N_moves_save_random[n]=np.copy(i)

            break

        i=i+1                               # UPDATE THE COUNTER



# AS YOU SEE, THE PERFORMANCE OF A RANDOM AGENT ARE NOT GREAT, SINCE THE MAJORITY OF THE POSITIONS END WITH A DRAW 
# (THE ENEMY KING IS NOT IN CHECK AND CAN'T MOVE)

print('Random_Agent, Average reward:',np.mean(R_save_random),'Number of steps: ',np.mean(N_moves_save_random))


#%%
## Let's define Adam, see notebook Adam's playgroud for more information

class Adam:

    def __init__(self, Params, beta1):
        
        N_dim=np.shape(np.shape(Params))[0] # It finds out if the parameters given are in a vector (N_dim=1) or a matrix (N_dim=2)
        
        # INITIALISATION OF THE MOMENTUMS
        if N_dim==1:
               
            self.N1=np.shape(Params)[0]
            
            self.mt=np.zeros([self.N1])
            self.vt=np.zeros([self.N1])
        
        if N_dim==2:
            
            self.N1=np.shape(Params)[0]
            self.N2=np.shape(Params)[1]
        
            self.mt=np.zeros([self.N1,self.N2])
            self.vt=np.zeros([self.N1,self.N2])
        
        # HYPERPARAMETERS OF ADAM
        self.beta1=beta1
        self.beta2=0.999
        
        self.epsilon=10**(-8)
        
        # COUNTER OF THE TRAINING PROCESS
        self.counter=0
        
        
    def Compute(self,Grads):
                
        self.counter=self.counter+1
        
        self.mt=self.beta1*self.mt+(1-self.beta1)*Grads
        
        self.vt=self.beta2*self.vt+(1-self.beta2)*Grads**2
        
        mt_n=self.mt/(1-self.beta1**self.counter)
        vt_n=self.vt/(1-self.beta2**self.counter)
        
        New_grads=mt_n/(np.sqrt(vt_n)+self.epsilon)
        
        return New_grads

beta1=0.9 # First order momentum for Adam

#%%
# INITIALISE THE PARAMETERS OF YOUR NEURAL NETWORK AND...
# PLEASE CONSIDER TO USE A MASK OF ONE FOR THE ACTION MADE AND ZERO OTHERWISE IF YOU ARE NOT USING VANILLA GRADIENT DESCENT...
# WE SUGGEST A NETWORK WITH ONE HIDDEN LAYER WITH SIZE 200. 

import numpy.matlib 
S,X,allowed_a=env.Initialise_game()
N_a=np.shape(allowed_a)[0]   # TOTAL NUMBER OF POSSIBLE ACTIONS
print(np.shape(allowed_a))
N_in=np.shape(X)[0]    ## INPUT SIZE
N_h=200                ## NUMBER OF HIDDEN NODES


## INITALISE YOUR NEURAL NETWORK...


# HYPERPARAMETERS SUGGESTED (FOR A GRID SIZE OF 4)

epsilon_0 = 0.2     # STARTING VALUE OF EPSILON FOR THE EPSILON-GREEDY POLICY
beta = 0.00005      # THE PARAMETER SETS HOW QUICKLY THE VALUE OF EPSILON IS DECAYING (SEE epsilon_f BELOW)
gamma = 0.85        # THE DISCOUNT FACTOR
eta = 0.0035        # THE LEARNING RATE

N_episodes = 5000 # THE NUMBER OF GAMES TO BE PLAYED 

# SAVING VARIABLES
R_save = np.zeros([N_episodes, 1])
N_moves_save = np.zeros([N_episodes, 1])

#weight initialise
#W1 = np.random.randn(N_in, N_h)/1000
#W2 = np.random.randn(N_h,N_a)/1000

W1 = np.random.uniform(0,1,(N_h, N_in))
W2 = np.random.uniform(0,1,(N_a, N_h))

# The following normalises the random weights so that the sum of each row =1
W1 = np.divide(W1,np.matlib.repmat(np.sum(W1,1)[:,None],1,N_in))
W2 = np.divide(W2,np.matlib.repmat(np.sum(W2,1)[:,None],1,N_h))

bias_W1 = np.zeros((N_h,))
bias_W2 = np.zeros((N_a,))

print("W",W1,W2)
print("bias",np.shape(bias_W1),bias_W1,bias_W2)
x1=np.dot(W1,X)+bias_W1
print("X",np.shape(X),np.shape(x1))
print(np.shape(np.dot(W2,x1)+bias_W2))
#%%
def EpsilonGreedy_Policy(Qvalues, epsilon):
    
    N_a=np.shape(Qvalues)[0]

    rand_value=np.random.uniform(0,1)

    rand_a=rand_value<epsilon
    a=np.where(allowed_a==1)[0]
    Qvalues_allowed=np.copy(Qvalues[a])
    if rand_a==True:
        
        a_agent=np.random.permutation(a)[0]

    else:
        '''
        sort_a=np.argsort(Qvalues)[::-1]
        while a_item in sort_a:
            if a_item in a:
                return a_item
        '''
        a1=np.argmax(Qvalues_allowed)
        a_agent=np.copy(a[a1])
            
    return a_agent

#%%
# TRAINING LOOP BONE STRUCTURE...
# I WROTE FOR YOU A RANDOM AGENT (THE RANDOM AGENT WILL BE SLOWER TO GIVE CHECKMATE THAN AN OPTIMISED ONE, 
# SO DON'T GET CONCERNED BY THE TIME IT TAKES), CHANGE WITH YOURS ...

for n in range(N_episodes):

    epsilon_f = epsilon_0 / (1 + beta * n)   ## DECAYING EPSILON
    Done=0                                   ## SET DONE TO ZERO (BEGINNING OF THE EPISODE)
    i = 1                                    ## COUNTER FOR NUMBER OF ACTIONS
    
    S,X,allowed_a=env.Initialise_game()      ## INITIALISE GAME
    print("N_episodes",n)                    ## REMOVE THIS OF COURSE, WE USED THIS TO CHECK THAT IT WAS RUNNING
    
    
    while Done==0:                           ## START THE EPISODE
        
        
        ## THIS IS A RANDOM AGENT, CHANGE IT to SARSA with value function approximation
        
        #a,_=np.where(allowed_a==1)
        #a_agent=np.random.permutation(a)[0]
        h1 = np.dot(W1,X)+bias_W1

        # Apply the sigmoid function
        x1 = 1/(1+np.exp(-h1))
        
        Qvalues=np.dot(W2,x1)+bias_W2
        
        #print("Qvalues",np.shape(Qvalues),Qvalues)
                
        #S_next,X_next,allowed_a_next,R,Done=env.OneStep(a_agent)
        
        a_agent = EpsilonGreedy_Policy(Qvalues, epsilon_f)
        S_next,X_next,allowed_a_next,R,Done=env.OneStep(a_agent)
       # print("S",S,"action",a_agent,R,Done)
        
        ## THE EPISODE HAS ENDED, UPDATE...BE CAREFUL, THIS IS THE LAST STEP OF THE EPISODE
        if Done==1:
            print("Done",Done)
            # Compute the delta
            #delta=R-Qvalues[a]
            
            # Update the Qvalues
            delta=R-Qvalues[a_agent]
                
           
            
            W2[a_agent,:]=W2[a_agent,:]+eta*delta*x1.T
            bias_W2[a_agent] = bias_W2[a_agent]+eta*delta
            
            W1=W1+eta*delta*X.T
            bias_W1 = bias_W1+eta*delta
            #b1[a_agent,:]=b1[a_agent,:]+eta*delta*X
            #b2=b2+eta*delta*h1
            #save the R_save,N_moves_save
            R_save[n]=np.copy(R)
            N_moves_save[n]=np.copy(i)
            
            
            break
        
        # IF THE EPISODE IS NOT OVER...
        h2 = np.dot(W1,X_next)+bias_W1

        # Apply the sigmoid function
        x2 = 1/(1+np.exp(-h2))
        
        Qvalues2=np.dot(W2,x2)+bias_W2
        a_pri = EpsilonGreedy_Policy(Qvalues2, epsilon_f)
        
        # Compute the delta
        delta=R+gamma*Qvalues2[a_pri]-Qvalues[a_agent]
        
        # Update the weights
        #print(np.shape(W2[a_agent,:]),np.shape(delta),np.shape(Qvalues2[a_pri]))
        W2[a_agent,:]=W2[a_agent,:]+eta*delta*x1.T
        bias_W2[a_agent] = bias_W2[a_agent]+eta*delta
            
        W1=W1+eta*delta*X.T
        bias_W1 = bias_W1+eta*delta
            
        # NEXT STATE AND CO. BECOME ACTUAL STATE...     
        S=np.copy(S_next)
        X=np.copy(X_next)
        allowed_a=np.copy(allowed_a_next)
        
        i += 1  # UPDATE COUNTER FOR NUMBER OF ACTIONS
        print("action i",i)
        
        
    
#%%

# Let's create a copy of the parameters to be optimised using Adam
W1_A = np.copy(W1)
W2_A = np.copy(W2)
bias_W1_A = np.copy(bias_W1)
bias_W2_A = np.copy(bias_W2)

# Intialise Adam for the parameters
Adam_W1=Adam(W1_A,beta1)
Adam_W2=Adam(W2_A,beta1)
Adam_bias_W1=Adam(bias_W1_A,beta1)
Adam_bias_W2=Adam(bias_W2_A,beta1)

#%%
# TRAINING LOOP BONE STRUCTURE...
# I WROTE FOR YOU A RANDOM AGENT (THE RANDOM AGENT WILL BE SLOWER TO GIVE CHECKMATE THAN AN OPTIMISED ONE, 
# SO DON'T GET CONCERNED BY THE TIME IT TAKES), CHANGE WITH YOURS ...

for n in range(N_episodes):

    epsilon_f = epsilon_0 / (1 + beta * n)   ## DECAYING EPSILON
    Done=0                                   ## SET DONE TO ZERO (BEGINNING OF THE EPISODE)
    i = 1                                    ## COUNTER FOR NUMBER OF ACTIONS
    
    S,X,allowed_a=env.Initialise_game()      ## INITIALISE GAME
    print("N_episodes",n)                    ## REMOVE THIS OF COURSE, WE USED THIS TO CHECK THAT IT WAS RUNNING
    
    
    while Done==0:                           ## START THE EPISODE
        
        
        ## THIS IS A RANDOM AGENT, CHANGE IT to SARSA with value function approximation and ADAM
        
        #a,_=np.where(allowed_a==1)
        #a_agent=np.random.permutation(a)[0]
        h1 = np.dot(W1_A,X)+bias_W1_A

        # Apply the sigmoid function
        x1 = 1/(1+np.exp(-h1))
        
        Qvalues=np.dot(W2_A,x1)+bias_W2_A
        
        #print("Qvalues",np.shape(Qvalues),Qvalues)
                
        #S_next,X_next,allowed_a_next,R,Done=env.OneStep(a_agent)
        
        a_agent = EpsilonGreedy_Policy(Qvalues, epsilon_f)
        S_next,X_next,allowed_a_next,R,Done=env.OneStep(a_agent)
       # print("S",S,"action",a_agent,R,Done)
        
        ## THE EPISODE HAS ENDED, UPDATE...BE CAREFUL, THIS IS THE LAST STEP OF THE EPISODE
        if Done==1:
            print("Done",Done)
            # Compute the delta
            #delta=R-Qvalues[a]
            
            # Update the Qvalues
            delta=R-Qvalues[a_agent]
                
            aa = eta*Adam_W2.Compute(delta)*x1.T
            
            W2_A[a_agent,:]=W2_A[a_agent,:]+aa[a_agent]
            bias_W2_A[a_agent] = bias_W2_A[a_agent]+eta*Adam_bias_W2.Compute(delta)[a_agent]
            
            W1_A=W1_A+eta*Adam_W1.Compute(delta)*X.T
            bias_W1_A = bias_W1_A+eta*Adam_bias_W1.Compute(delta)
            #b1[a_agent,:]=b1[a_agent,:]+eta*delta*X
            #b2=b2+eta*delta*h1
            #save the R_save,N_moves_save
            R_save[n]=np.copy(R)
            N_moves_save[n]=np.copy(i)
            
            
            break
        
        # IF THE EPISODE IS NOT OVER...
        h2 = np.dot(W1_A,X_next)+bias_W1_A

        # Apply the sigmoid function
        x2 = 1/(1+np.exp(-h2))
        
        Qvalues2=np.dot(W2_A,x2)+bias_W2_A
        a_pri = EpsilonGreedy_Policy(Qvalues2, epsilon_f)
        
        # Compute the delta
        delta=R+gamma*Qvalues2[a_pri]-Qvalues[a_agent]
        
        # Update the weights
        #print(np.shape(W2[a_agent,:]),np.shape(delta),np.shape(Qvalues2[a_pri]))
        bb = eta*Adam_W2.Compute(delta)*x1.T
        W2_A[a_agent,:]=W2_A[a_agent,:]+bb[a_agent]
        bias_W2_A[a_agent] = bias_W2_A[a_agent]+eta*Adam_bias_W2.Compute(delta)[a_agent]
            
        W1_A=W1_A+eta*Adam_W1.Compute(delta)*X.T
        bias_W1_A = bias_W1_A+eta*Adam_bias_W1.Compute(delta)
            
        # NEXT STATE AND CO. BECOME ACTUAL STATE...     
        S=np.copy(S_next)
        X=np.copy(X_next)
        allowed_a=np.copy(allowed_a_next)
        
        i += 1  # UPDATE COUNTER FOR NUMBER OF ACTIONS
        print("action i",i)
        
        
    
    

