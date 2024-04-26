#概率
# %matplotlib inline
import torch
from torch.distributions import multinomial
# from d2l import torch as d2l

#采样
#fair_probs为概率
fair_probs = torch.ones([6]) / 6
print(multinomial.Multinomial(1, fair_probs).sample())
print(multinomial.Multinomial(10, fair_probs).sample())

counts = multinomial.Multinomial(10000, fair_probs).sample()
print(counts / 10000)  # 相对频率作为估计值


