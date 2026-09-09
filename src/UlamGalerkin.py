import numpy as np

def UlamGalerkin(D,ds=0.2):
    _,m    = D.shape                     # data resolution
    DT     = D.T 
    M      = np.zeros((m,m),dtype=int)   # counting matrix 
    ic     = 0                           # total cell counter
    H      = [np.floor(DT[0,:]/ds).astype(int).tolist()]  # hash list
    DD     = zip(DT[:-1,:],DT[1:,:])     # twin-snapshot matrix
    jj     = 0
    M[0,0] = 1
    for dj,di in DD:
        hi = np.floor(di/ds).astype(int).tolist()  # i-hash 
        hj = np.floor(dj/ds).astype(int).tolist()  # j-hash 
        if hi in H:
            ii = H.index(hi)             # get index ii
        else:
            H.append(hi)                 # add to hash list 
            ic += 1                      # update counter
            ii  = ic                     # new index 
        M[ii,jj] += 1                    # update counting matrix
        jj = ii
    Mc       = M[:ic+1,:ic+1]            # extract proper size 
    row_sums = Mc.sum(axis=1)
    P        = Mc/row_sums[:,np.newaxis] # make row-stochastic 
    return P,H

if __name__ == '__main__':

    n,m = 3,1000
    D   = np.random.random((n,m)) 
    P,H = UlamGalerkin(D,0.2)
    
    
