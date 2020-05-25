import torch
from torch.nn import Parameter as P
wei = P(torch.rand(1,10,100,5))
x = P(torch.rand(64,5))
y = torch.randint(0,5,[64])
wei = wei.expand(64,-1,-1,-1)
all = (wei - x.unsqueeze(1).unsqueeze(1)).norm(dim=-1)

mask = torch.zeros(size=(wei.size(0),wei.size(1)),dtype=torch.bool)

mask[torch.arange(0,64),y] = True


cha = all.gather(1,y.unsqueeze(1).unsqueeze(2).expand(-1,-1,100)).squeeze()
sum=0

cha_1 = all[mask.unsqueeze(2).expand(-1,-1,100)].reshape(mask.size(0),-1)

sum = (cha-cha_1).sum()
assert sum == 0
for i in range(0, 64):
    for j in range(0,100):
        sum += cha_1[i,j] - (wei[i,y[i],j,:] - x[i,:]).norm()
        if sum != 0:
            print(i,sum)

pull = cha_1.max(-1)[0]
mask_fan = mask[:] == False

cha_fan = all[mask_fan.unsqueeze(2).expand(-1,-1,100)].reshape(mask.size(0),-1,all.size(-1))


for i in range(0, 64):
    bias = 0
    for j in range(0, 10):
        if mask[i,j] == False:
            sum += (cha_fan[i,j-bias,:] - all[i,j,:]).sum()
            assert sum ==0
        else:
            bias = 1

push = cha_fan.min(-1)[0]

push_mask = push[push < pull.unsqueeze(1)]









