import torch

import torch.nn.functional as F



a=torch.tensor([-1.0,1,2,3,4])
b=F.relu(a)
c=F.leaky_relu(a)
d=torch.sigmoid(a)
e=torch.tanh(a)
print(b,c,d,e)