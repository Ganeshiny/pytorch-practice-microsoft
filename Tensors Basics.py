%matplotlib inline
import torch
import numpy as np


'''
Initializing tensors
'''

#From data 
data = [[1,2, 3],[3, 4, 5]]
x_data = torch.tensor(data)

#From a numpy array

np_array = np.array(data)
x_np = torch.from_numpy(np_array)

'''
np_array is a ndarray 

The returned tensor and the ndarray share the
same memory. Modifications to the tensor will 
be reflected in the ndarray and vicecersa. The 
returned tensor is not resizble

It currently accepts ndarray with dtypes 
of numpy.float64, numpy.float32, 
numpy.float16, numpy.complex64, 
numpy.complex128, numpy.int64,
numpy.int32, numpy.int16, numpy.int8, 
numpy.uint8, and numpy.bool.

'''

#Writing to a tensor created from a 
#read-only NumPy array is not 
#supported and will result 
#in undefined behavior.

#From another tensor

#The below retains the shape and data type of the arguement tensor
x_ones = torch.ones_like(x_data) # retains the properties of x_data
print(f"Ones Tensor: \n {x_ones} \n")

#The below example shows overidding of datatype of x_data
x_rand = torch.rand_like(x_data, dtype=torch.float) # overrides the datatype of x_data
print(f"Random Tensor: \n {x_rand} \n")

#With random or constant values 

shape = (2, 3, ) #tuple, (#rows, #cols, )
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)

'''
Tensor attributes
'''
#shape
#datatype
#device on which they are stored

tensor = torch.rand(3,4)

print(f"Shape of tensor: {tensor.shape}")
print(f"Datatype of tensor: {tensor.dtype}")
print(f"Device tensor is stored on: {tensor.device}")


print(torch.cuda.is_available())

#Moving the tensor to the GPU
if torch.cuda.is_available():
    tensor = tensor.to('cuda')

#numPy like indexing and slicing 

'''
we are going to use tensorAPI

'''
tensor = torch.ones(4, 4)
'''
tensor[0] : returns the first row
tensor[:, 0] : Returns the first column
tensor[..., -1] : Returns the last column

tensor[:, 1] = 0 makes allt the values in the
second column 0 

'''

#Joining tensors

'''Arithmetic operations'''

#Matrix multiplication (3)

#Element wise product

'''#--"'''

# Single element tensors

#In place operations

'''
Tensor to numpy bridge
'''

#Tensor to numpy

#Numpy to Tensor










