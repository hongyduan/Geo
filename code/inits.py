import math
import torch
def uniform(size, tensor):
    bound = 1.0 / math.sqrt(size)
    if tensor is not None:
        tensor.data.uniform_(-bound, bound)

def mul_def(entity_em_matrix, concept_em_matrix, em_dim):
    final_matrix = torch.zeros((entity_em_matrix.shape[0], concept_em_matrix.shape[0], em_dim), dtype=torch.float64) # 1545*106*500
    for i in range(entity_em_matrix.shape[0]): # 1545
        i_em = entity_em_matrix[i,:]
        for j in range(concept_em_matrix.shape[0]): # 106
            j_em = concept_em_matrix[j,:] # 1*500
            tmp_em = (i_em.mul(j_em)).mul(i_em) # 1*500
            # tmp_em = i_em * j_em * i_em
            final_matrix[i,j,:] = tmp_em
    return final_matrix
