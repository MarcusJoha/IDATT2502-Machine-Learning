import torch
import matplotlib.pyplot as plt
import csv

file = open("files/day_head_circumference.csv")

reader = csv.reader(file, delimiter=',')

header = []
header = next(reader)

x = []
y = []

for row in reader:
    x.append(float(row[0]))
    y.append(float(row[1]))

# print(x)

x_train = torch.tensor(x).reshape(-1,1)
y_train = torch.tensor(y).reshape(-1,1)

    

class NonLinearRegressionModel:
    def __init__(self):
        self.W = torch.tensor([[0.0]], requires_grad=True)
        self.b = torch.tensor([[0.0]], requires_grad=True)
        
    def sigmoid(self, exp):
        return 1 / (1 + torch.exp(-exp))
        
    def f(self, x):
        sig = 20 * self.sigmoid(x @ self.W + self.b) + 31
        return sig
    
    def loss(self, x, y):
        return torch.mean(torch.square(self.f(x) - y))
    
model = NonLinearRegressionModel()

optimizer = torch.optim.Adam([model.W, model.b], 0.001)

for epoch in range(10000):
    loss = model.loss(x_train, y_train)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    if epoch%500==0:
        print("epoch: {}; loss: {}".format(epoch, loss))


# Print model variables and loss
print("W = %s, b = %s, loss = %s" % (model.W, model.b, model.loss(x_train, y_train)))


xs = torch.linspace(torch.min(x_train).item(), torch.max(x_train).item(), 100).reshape(100, 1)
ys = model.f(xs)

# Visualize result
plt.plot(x_train, y_train, 'o', label='$(x^{(i)},y^{(i)})$')
plt.xlabel('x')
plt.ylabel('y')
plt.plot(xs, ys.detach())
plt.legend()
plt.show()