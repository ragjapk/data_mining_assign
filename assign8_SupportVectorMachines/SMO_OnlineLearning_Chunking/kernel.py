import numpy as np


def kernel_function(kernel,i,j):
    if kernel == "Linear" :
        kernel_value = np.dot(i,j)
    elif kernel == "Polynomial" :
        kernel_value = ((np.dot(i,j))**2)
    return kernel_value
 
    
def get_kernel_matrix(data,n,kernel) :
    kernel_matrix = np.zeros([n,n])
    for i in range(n):
        for j in range(n):
            if kernel == "Linear" :
                kernel_matrix[i,j] = kernel_function(kernel,data[i],data[j])
            elif kernel == "Polynomial":
                kernel_matrix[i,j] = kernel_function(kernel,data[i],data[j])
    return kernel_matrix
