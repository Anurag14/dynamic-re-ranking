import numpy as np
import torch
import time
"""
    The following is a function f that contains the constraints for the objective to be optimizied
    mathematically defined as f(Z,Q,E,W,X,A,mu,Y1,Y2)= (mu/2)(||WX-WXZ-E+(Y_1/mu)||_F^{2} + ||Z-Q+(Y_2/mu)||_F^{2})
"""
def grad_f(Z,Q,E,X,A,mu,Y1,Y2):
    abs1=X-torch.mm(X,Z)-E+Y1/mu
    abs2=Z-Q+Y2/mu
    return torch.mm(torch.transpose(X,0,1),abs1)-abs2

"""
    Computes the laplacian matrix for a given graph L_A= D_A -A where D_A is the degree matrix of A
"""
def laplacian_matrix(A):
    D_A=torch.zeros(A.shape[0],A.shape[0])
    for i in range(A.shape[0]):
        D_A[i][i]=A[i].nonzero().size()[0] # finds the number of outgoing egdes from the node aka its degree.
    return D_A - A


"""
    soft thresholding operator with definiton
    \hat{f(y)}= y+ \tau if y<-\tau
             0       if -\tau<y<\tau
             y- \tau if y>\tau
"""
def L21_operator(M,N,sigma):
    for i in range(M.shape[1]):
        column_norm=torch.norm(N[:,i])
        M[:,i]=max(1-sigma/column_norm,0)*N[:,i]
    return M

"""
    L21 minimization for the proximal operator  for the update rule of error matrix E
"""
def soft_operator(y,sigma):
    return torch.add(torch.clamp(y+sigma,max=0),torch.clamp(y-sigma,min=0))
"""
    Functions u_vector and update_A are for the update rule of A which is row wise updated.
"""
def u_vector(Q,parameters):
    u=torch.zeros(Q.shape[0],Q.shape[0],requires_grad=False)
    for i in range(Q.shape[0]):
        for j in range(i+1,Q.shape[1]):
            u[i][j]=0.5*(torch.norm(Q[:,i]-Q[:,j]))*parameters['gamma']
            u[j][i]=u[i][j]
    return u

def u_vector_mesh(Q,parameters):
    cols = Q.shape[1]
    # equivalent to generating a meshgrid
    c1 = [i for i in range(cols) for _ in range(cols)]
    c2 = [i for _ in range(cols) for i in range(cols)]
    return 0.5 * torch.norm(Q[:,c1] - Q[:,c2], dim=0).reshape(cols, cols) * parameters['gamma']


def u_vector_new(Q,parameters):
    rows, cols = Q.shape[0], Q.shape[1]
    c1 = [i for i in range(cols) for _ in range(cols)]
    c2 = [i for _ in range(cols) for i in range(cols)]
    u = 0.5 * torch.norm(Q[:,c1] - Q[:,c2], dim=0).reshape(cols, cols) * parameters['gamma']
    return torch.nn.functional.pad(u, (0, rows-cols, 0, rows-cols), 'constant', 0)

def update_A_new(A,Q,parameters):
    u = u_vector(Q,parameters)
    print("u updated, now going for A update")
    print(Q.shape,u.shape)
    for j in range(A.shape[1]):
        temp,_=u[:,j].topk(parameters['epsilon'],largest=False)
        _,indicies=u[:,j].topk(A.shape[1]-parameters['epsilon'])
        res=torch.sum(temp)
        res=(1+res)/parameters['epsilon']
        A[:,j]=torch.clamp(res*torch.ones_like(u[:,j])-u[:,j],min=0)
        A[indicies,j]=0
    return A

"""
    The following is a conversion function for the convex optimization procedure being employed
    for building the dynamic affinity matrix A.
    Input: LOMO feature matrix X ( should be reduced to an appropriate dimension using PCA)
    Intermediate Variables: k (denotes the number of iterations)
                            parameters {dimension of W:,alpha:,beta:,gamma:,lambda3:,lambda4:,tau:,epsilon:,eta:} //dict with given entries             
                            Set Z = Q = A = Y2 = 0, E = Y1 = 0.                           
                            mu=0.1, mu_max=10**10, rho=1.1                                  
    Output: Z, Q, A, W, E
    form of objective is alpha||Z||_1 + beta||E||_{2,1} + gamma tr(QL_AQ^{T}) + lambda ||A||_{F} + thetha ||W||_{2,1}
"""

def convex_update(X,parameters,tensor_dict,device):
    Z,Q,A,Y2,E,Y1 = tensor_dict['Z'],tensor_dict['Q'],tensor_dict['A'],tensor_dict['Y2'],tensor_dict['E'],tensor_dict['Y1']
    mu, mu_max, rho, k = tensor_dict['mu'], 10**10, 1.5, tensor_dict['k']
    del(tensor_dict)
    OptimizationLoss=0
    while k<=parameters['iterations']:
        #update Z
        temp = Z + grad_f(Z,Q,E,X,A,mu,Y1,Y2)/parameters['eta']
        Z = soft_operator(temp,parameters['alpha']/(parameters['eta']*mu))
        del(temp)
        #update Q
        L_A=laplacian_matrix(A)
        part1 = mu*Z + Y2
        part2 = torch.inverse(mu*torch.eye(X.shape[1]) + parameters['gamma']*(L_A+torch.transpose(L_A,0,1)))
        Q = torch.mm(part1,part2)
        del(part1)
        del(part2)
        del(L_A)
        #update E
        aux = X-torch.mm(X,Z)+Y1/mu
        E=L21_operator(E,aux,parameters['beta']/mu)
        del(aux)
        #update A
        print("Now updating A")
        A = update_A_new(A,Q,parameters)
        Y1 = Y1+mu*(X-torch.mm(X,Z)-E)
        Y2 = Y2+mu*(Z-Q)
        mu=min(mu_max,rho*mu)
        StepLoss=torch.norm(X-torch.mm(X,Z))
        OptimizationLoss+=StepLoss
        print(k)
        print("Z:\n",Z)
        print("Q:\n",Q)
        print("A:\n",A)
        print("Step Loss:",StepLoss)
        print("Z Q diff:",torch.norm(Z-Q))
        k+=1
    print("otpimization loss:",OptimizationLoss/parameters['iterations'])
    tensor_dict={'Z':Z,'Q':Q,'A':A,'E':E,'Y1':Y1,'Y2':Y2,'mu':mu,'k':k}
    return tensor_dict


