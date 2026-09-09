
# %%
# %%
import numpy as np
def ulamGalerkin(Data,Nd=5,sim=np.inf,sample=1,ulam_scaled=False,ulam_dt=0.05):        #Data here are of size (Time x physical variables)(Rows are time while coulmns are physical variables)
    x=Data.copy()      
    mini = np.min(x, axis=0)        #Extract minimum value of all physical variables 
    maxi = np.max(x, axis=0)        #Extract maximum value of all physical variables
    delta=(maxi+10e-12-mini)/Nd            #Discretize the state space into cells
    xidx=np.floor((x-mini)/delta).astype(int) #Assign each point in the data to a cell in the state space
    unique_rows, inverse_indices = np.unique(xidx, axis=0, return_inverse=True)  #unique_rows of x = boxnumbers or box id, inverse_indices are snapshots id 
    bx_to_p_dict={}
    p_to_bx_dict=np.array([tuple(col) for col in xidx])                          #This a list of boxes of each snapshot p_to_bx_dict[0]=[1,1]
    #M[0,0]=1                                                                  #first snapshot assignment
    for i in range(len(unique_rows)):
        bx_to_p_dict[tuple(unique_rows[i])]= np.where(inverse_indices == i)[0]

    if ulam_scaled:
        U_local=np.zeros(len(unique_rows))
    M=np.zeros((len(unique_rows),len(unique_rows)))     #start the markov matrix
    if np.isfinite(sim):
        boundary = set(np.arange(sim, len(inverse_indices), sample))
        #print(boundary)
    else:
        boundary = set()
    for i in range(len(inverse_indices)-1):
        if i+1 in boundary:
            continue
        if ulam_scaled:
            M[inverse_indices[i+1],inverse_indices[i]]=M[inverse_indices[i+1],inverse_indices[i]]+1
            if inverse_indices[i+1]==inverse_indices[i]:
                        U_local[inverse_indices[i]]+=np.linalg.norm(x[i+1]-x[i])
        else:
            M[inverse_indices[i+1],inverse_indices[i]]=M[inverse_indices[i+1],inverse_indices[i]]+1
    if ulam_scaled:
        U_local=U_local/M.sum(axis=0)
        no_self_transition_idx=U_local==0
        dt_U_local_inv_dx=U_local*(ulam_dt/np.linalg.norm(delta))
        dt_U_local_inv_dx[no_self_transition_idx]=1
        idx = np.diag_indices_from(M)
        M[idx] *= dt_U_local_inv_dx
        M=np.floor(M)

    M = M / M.sum(axis=0,keepdims=True)  #Normalize over the rows where sum of columns equal to 1 (j>>>all i)
    if ulam_scaled:
    
    #print("changed")
        return M,bx_to_p_dict,p_to_bx_dict,mini,maxi,delta,dt_U_local_inv_dx   #Return markov matrix, a dictionary of box:[snapshots,,,], a list of snapshot assignment to boxes P:[box,,,,]
    return M,bx_to_p_dict,p_to_bx_dict,mini,maxi,delta

if __name__ == '__main__':

    n,m = 3,1000
    D   = np.random.random((n,m)) 
    P,H,B = ulamGalerkin(D.T,5)
    #print(B)

# %%


# %%


# %%



