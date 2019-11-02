import math
import torch
def uniform(size, tensor):
    bound = 1.0 / math.sqrt(size)
    if tensor is not None:
        tensor.data.uniform_(-bound, bound)

def mul_def(entity_em_matrix, concept_em_matrix, em_dim):  # 6178*200;  6178*150*200;  200
    final_matrix = torch.zeros((entity_em_matrix.shape[0], concept_em_matrix.shape[1], em_dim)) # 6178*  150*200
    for i in range(entity_em_matrix.shape[0]): # 6178
        i_em = entity_em_matrix[i,:]

        final_matrix[i,:,:] = (i_em.mul(concept_em_matrix[i,:,:])).mul(i_em)


    return final_matrix
