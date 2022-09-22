import torch
import numpy as np


# Initializing a Tensor: Directly from data

data = [[1, 2], [3, 4]]  # Some random 2D list
print(f'Some random data: {data}')
x_data = torch.tensor(data)  # Convert the 2D list into a tensor
print(f'Same random data as a torch tensor: {x_data}')

np_array = np.array(data)  # Convert the list into a NumPy Array
print(f'Random NumPy Array: {np_array}')
x_np = torch.from_numpy(np_array)  # Convert NumPy array into tensor
print(f'Random NumPy Array as a torch tensor: {x_np}')

x_ones = torch.ones_like(x_data)
print(f'Ones Tensor: \n {x_ones} \n')

x_rand = torch.rand_like(x_data, dtype=torch.float)  # Overrides the datatype of x_data
print(f'Random Tensor: \n {x_rand}\n')


# Tensors with random or constant values

shape = (2, 3,)
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)

print(f'Random Tensor: \n {rand_tensor} \n')
print(f'Ones Tensor: \n {ones_tensor} \n')
print(f'Zeros Tensor: \n {zeros_tensor} \n') 


# Attributes of a Tensor

tensor = torch.rand(3, 4)
print(f'Some random tensor: \n {tensor}\n')

print(f'Shape of tensor: {tensor.shape}')
print(f'Datatype of tensor: {tensor.dtype}')
print(f'Device tensor is stored on: {tensor.device}')



'''
Operations on Tensor

By default, tensors are created on the CPU. We need to explicitly move 
tensors to the GPU using .to method (after checking for GPU availability).
Keep in mind that copying large tensors across devices can be expensive in
terms of time and memory!
'''

if torch.cuda.is_available():
    print('cuda device is available, moving tensor to GPU')
    tensor = tensor.to('cuda')
    print(f'GPU tensor: \n {tensor}')

tensor = torch.ones(4, 4)
print(f'Tensor of ones: \n {tensor}')
print(f"First row: {tensor[0]}")
print(f"First column: {tensor[:, 0]}")
print(f"Last column: {tensor[..., -1]}")
tensor[:,1] = 0
print(tensor)

# Joining tensors, you can user torch.cat to concatenate a sequence of tensors
t1 = torch.cat([tensor, tensor, tensor], dim=1)
print(f'Concatenated tensor: \n {t1}')


# This computes the matrix multiplication between two tensors. y1, y2, y3 will
# have the same value

y1 = tensor @ tensor.T
y2 = tensor.matmul(tensor.T)

y3 = torch.rand_like(y1)
torch.matmul(tensor, tensor.T, out=y3)

