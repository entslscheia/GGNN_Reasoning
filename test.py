import torch
from torch.autograd import Variable
import numpy as np
import torch.nn as nn

# a = torch.IntTensor(np.array([[2.7,3,4]))
# print(a.shape)
# a_numpy = np.array([2,3,4])
# print(a.numpy() == a_numpy)

# x = torch.randn(3)
# x = Variable(x, requires_grad=True)
# print(x.data)
# y = x * 2
# print(y)
# y.backward(torch.Tensor([1,0.01,0.1]))
# print(x.grad)

# class TestNN(nn.Module):
#     def __init__(self):
#         super(TestNN, self).__init__()  # this is necessaey
#         # self.w = Variable(torch.FloatTensor([1,2,3]), requires_grad=True)
#         self.w = torch.nn.Parameter(torch.FloatTensor([1,2,3]))
#         print("modules: " + str(list(self.modules())))
#
#     def forward(self, x):
#         return self.w * x
#
# model = TestNN()
# #x = torch.randn(3, requires_grad=True)
# x = torch.tensor([1,1,1]).float()
# print(x.requires_grad)
# print(x.data)
#
# print(model.parameters())
# for param in model.parameters():
#     print(param)
#
# optimizer = torch.optim.SGD(model.parameters(), lr=0.3)
# for i in range(10):
#     print('x' + str(x.grad))
#     output = model(x)
#     # print(output)
#     target = torch.Tensor(torch.zeros(3))
#     criterion = nn.MSELoss()
#     loss = criterion(output, target)
#     # model.zero_grad()   # this seems to be an alternative solution for the following sentence
#     optimizer.zero_grad()  # necessary every time update the weights
#     loss.backward()
#     optimizer.step()
#     print(str(loss.data) + ",  " + str(model.w.data))

# m = nn.Linear(20, 30)
# input = torch.randn(100, 5, 20)
# output = m(input)
# print(output.size())

# example of sequential
# class TestSeq(nn.Module):
#     def __init__(self):
#         super(TestSeq, self).__init__()
#         self.seqNN = nn.Sequential(nn.Linear(2, 3),
#                                    nn.Linear(3, 4),
#                                    nn.Sigmoid())
#     def forward(self, x):
#         return self.seqNN(x)
#
# x = torch.FloatTensor([1,2])
# nn = TestSeq()
# print(nn(x))

# example of use of contiguous
# x = torch.FloatTensor([[1,2,3], [4,5,6]])
# print(x.is_contiguous())
# y = x.t()
# print(y.is_contiguous())
# print(y.contiguous().is_contiguous())

#  stack example
# print(torch.Tensor([1]))
# a = [torch.Tensor([1]), torch.Tensor([2]), torch.Tensor([3])]
# print(a)
# print(torch.stack(a))   #  in fact it doesn't change the structure, just transform a list into a tenor of the same shape

embed = nn.Embedding(2, 4, sparse=True)
embed.weight.data.copy_(torch.Tensor([[1,2,3,4],[2,3,4,5]]))
# Then turn the word index into actual word vector
# vocab = {"a": 0, "b": 1}
# word_indexes = [vocab[w] for w in ["a", "b"]]
# word_vectors = embed(torch.LongTensor(word_indexes))
#print(word_vectors)
class embedTest(nn.Module):   # test how to update embedding
    def __init__(self):
        super(embedTest, self).__init__()
        self.embed = embed
        self.attention = nn.Sigmoid()
        self.out = nn.Tanh()

    def forward(self, x):
        if sum(x).numpy() > 0:
            vec = self.embed(torch.LongTensor([0]))
            vec = vec.view([2,2])
            return self.out(self.attention(vec * x))
        else:
            vec = self.embed(torch.LongTensor([1]))
            vec = vec.view([2, 2])
            return self.out(self.attention(vec * x))
net = embedTest()
x = torch.Tensor([1,1])
for idx, m in enumerate(net.modules(), 0):
    print(idx, '->', m)
for idx, p in enumerate(net.parameters(), 0):
    print(idx, ': ', p)
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(net.parameters(), lr=0.05)
for i in range(10):
    # x = torch.randn(2)
    optimizer.zero_grad()
    output = net(x)
    # print(output)
    print(embed(torch.LongTensor([0, 1])))
    loss = criterion(torch.zeros((2)), output)
    loss.backward()
    optimizer.step()

# a = [torch.randn(3) for k in range(5)]
# print(a)
# a[2] += torch.Tensor([1, 1, 1])R
# print(a)
# print(torch.stack(a))
