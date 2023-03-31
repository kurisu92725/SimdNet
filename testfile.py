import torch
# import torch_geometric
import torch.nn.functional as F
print(torch.cuda.is_available())

# print(torch_geometric.__version__)
a=torch.tensor([-1.0,1,2,3,4])
b=F.relu(a)
c=F.leaky_relu(a)
d=torch.sigmoid(a)
e=torch.tanh(a)
print(b,c,d,e)