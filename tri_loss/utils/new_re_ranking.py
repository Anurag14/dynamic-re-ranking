"""
Created on Mon Nov 11 14:46:56 2019

@author: anurag singh

- This version accepts distance matrix instead of raw features.
- The difference of `/` division between python 2 and 3 is handled.
- numpy.float16 is replaced by numpy.float32 for numerical precision.
"""

"""
K-reciprocal code is inspired from 
CVPR2017 paper:Zhong Z, Zheng L, Cao D, et al. Re-ranking Person Re-identification with k-reciprocal Encoding[J]. 2017.
url:http://openaccess.thecvf.com/content_cvpr_2017/papers/Zhong_Re-Ranking_Person_Re-Identification_CVPR_2017_paper.pdf
Matlab version: https://github.com/zhunzhong07/person-re-ranking
"""

"""
API

q_g_dist: query-gallery distance matrix, numpy array, shape [num_query, num_gallery]
q_q_dist: query-query distance matrix, numpy array, shape [num_query, num_query]
g_g_dist: gallery-gallery distance matrix, numpy array, shape [num_gallery, num_gallery]

k1, k2, lambda_value: parameters, the original paper is (k1=20, k2=6, lambda_value=0.3)

Returns:
  final_dist: re-ranked distance, numpy array, shape [num_query, num_gallery]
"""

import torch
import numpy as np
from patch_optimization import convex_update

def graph_re_ranking(features, q_g_dist, k1=20, k2=6, lambda_value=0.3):

    # The following naming, e.g. gallery_num, is different from outer scope.
    # Don't care about it.
    #append all in A_big 
    big_A=[]
    alpha_crop_gallery=50
    parameters={'alpha': 0.1,'beta':0.1,'gamma':10,'lambda':1,'epsilon':k1+1,'eta':0,'iterations':50}
    for probe_index in range(len(q_g_dist)):
        index_of_features=[]
        index_of_features=np.argpartition(q_g_dist[probe_index],alpha_crop_gallery_probe+1)
        with torch.no_grad():
            X=torch.transpose(features[index_of_features[:all_crop_gallery_probe+1]],0,1)
            parameters['eta']=torch.sum(X**2)
            A=convex_update(X,parameters)
            big_A.append(A.detach().numpy())
    for A in big_A:
        V = np.zeros_like(A).astype(np.float32)
        initial_rank = np.argsort(A).astype(np.int32)
        forward_k_neigh_index = initial_rank[:k1+1,0]
        backward_k_neigh_index = initial_rank[:k1+1,forward_k_neigh_index]
        f0 = np.where(backward_k_neigh_index==0)[1]
        k_reciprocal_index = forward_k_neigh_index[f0]
        k_reciprocal_expansion_index = k_reciprocal_index
        for j in range(len(k_reciprocal_index)):
            candidate = k_reciprocal_index[j]
            candidate_forward_k_neigh_index = initial_rank[:int(np.around(k1/2.))+1,candidate]
            candidate_backward_k_neigh_index = initial_rank[:int(np.around(k1/2.))+1,candidate_forward_k_neigh_index]
            fi_candidate = np.where(candidate_backward_k_neigh_index == candidate)[1]
            candidate_k_reciprocal_index = candidate_forward_k_neigh_index[fi_candidate]
            if len(np.intersect1d(candidate_k_reciprocal_index,k_reciprocal_index))> 2./3*len(candidate_k_reciprocal_index):
                k_reciprocal_expansion_index = np.append(k_reciprocal_expansion_index,candidate_k_reciprocal_index)
    
        k_reciprocal_expansion_index = np.unique(k_reciprocal_expansion_index)
        weight = np.exp(-np.reciprocal(A[k_reciprocal_expansion_index,0]))
        V[k_reciprocal_expansion_index,0] = 1.*weight/np.sum(weight)
        if k2 != 1:
            V_qe = np.zeros_like(V,dtype=np.float32)
            V_qe[:,0] = np.mean(V[initial_rank[:k2,0],:],axis=0)
            V = V_qe
        del initial_rank
        A[:,0] = V[:,0]*(1-lambda_value) + A[:,0]*lambda_value
        del V
        final_rank_list=np.argsort(A[:,0])
        print(final_rank_list)
    return final_cmc,final_map
