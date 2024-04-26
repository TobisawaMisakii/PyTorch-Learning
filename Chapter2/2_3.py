import torch as t
from torch import float32

# #向量
# x = t.arange(4)
# print(x[3])
# print(x.shape)
#
# #矩阵
# X = t.arange(12).reshape(3,4)
# print(X, end='\n')
# print(X.shape)
# print(X[2][3])
# #矩阵转置
# print(X.T)
#
# #张量
# M = t.arange(24).reshape(-1,3,4)
# print(M)
# #element-wise
# print(M*M)#Hadamard product
# print(3+M)
#
# #降维
# print(M.sum())
# print(M.sum(axis=0))
# print(M.sum(axis=1))
# print(M.sum(axis=2))

# M = t.arange(24, dtype=float).reshape(2,3,4)
# print(M.mean())
# print(M.sum()/M.numel())
# print(M.mean(axis=0))
# print(M.sum(axis=0)/M.shape[0])
# print(M.mean(axis=1))
# print(M.sum(axis=1)/M.shape[1])
# print(M.mean(axis=2))
# print(M.sum(axis=2)/M.shape[2])

#非降维求和
#保持维度不变
M = t.arange(24, dtype=float).reshape(2,3,4)
print(M)
sum_M = M.sum(axis=2, keepdims=True)
print(sum_M)
print(M.sum(axis=2))

print(M / sum_M)

#cumsum 沿某个轴计算 M元素的累积总和
print(M.cumsum(axis=0))
print(M.cumsum(axis=2))

#dot product
x = t.arange(4, dtype=t.float32)
y = t.ones(4, dtype=t.float32)
print(x)
print(y)
print(t.dot(x,y))

#Matric * vector
MatricA = t.arange(24,dtype=t.float32).reshape(4, 6)
print(MatricA)
print(MatricA.shape)
print(y.shape)
print(t.mv(MatricA.T, y))

#Matric * Matric
MatricB = t.ones(24, dtype=float32).reshape(4, 6)
print(MatricB.T)
print(MatricA)
print(t.mm(MatricA, MatricB.T))

#Norm L2范数
print(t.norm(MatricA))
print(t.norm(MatricB))

