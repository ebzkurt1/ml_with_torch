import torch


x = torch.ones(5)  # Input tensor
y = torch.zeros(3)  # expected out
w = torch.randn(5, 3, requires_grad=True)
b = torch.randn(3, requires_grad=True)
z = torch.matmul(x, w)+b
loss = torch.nn.functional.binary_cross_entropy_with_logits(z, y)

print(f'Gradient function for z = {z.grad_fn}')
print(f'Gradient funtion for loss = {loss.grad_fn}')


# Computing Gradients

loss.backward()
print(w.grad)
print(b.grad)
