import torch

input = torch.tensor([1,2,3,4,5,6])
print(torch.where(input>=5)[0])



exit()


true = torch.tensor([[1, 0, 0, 0],[0, 1, 0, 0]])
print(torch.where(true > 0, 1.0, 0.3))
exit()


batch = torch.tensor([
    [[1, 1, 1, 1], [2, 2, 2, 2]],
    [[3, 3, 3, 3], [4, 4, 4, 4]],
    [[5, 5, 5, 5], [6, 6, 6, 6]],
])  # shape:[3, 2, 4]

idx = torch.tensor([[1, 0], [0, 1], [1, 0]])  # shape:[3, 2]
idx = idx.unsqueeze(1)  # shape:[3,1,2]
batch_n = torch.matmul(idx, batch)

print(batch_n.shape) # shape:[3, 1, 4]
print(batch_n + batch) # shape:[3, 2, 4]