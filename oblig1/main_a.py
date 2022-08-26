import torch
import matplotlib.pyplot as plt
import csv

file = open('files/length_weight.csv')

csvreader = csv.reader(file, delimiter =',')

header = []
header = next(csvreader)

x_train = []
y_train = []

for row in csvreader:
    x_train.append(float(row[0])) # forandrer denne senere
    y_train.append(float(row[1])) # når jeg har funnet tall buggen

# print(x_train)
x_train_tensor = torch.tensor(x_train).reshape(-1, 1)
y_train_tensor = torch.tensor(y_train).reshape(-1, 1)


class LinearRegresionModel:
    def __init__(self):
        self.W = torch.tensor([[0.0]], requires_grad=True)
        self.b = torch.tensor([[0.0]], requires_grad=True)

    def f(self, x):
        return x @ self.W + self.b # @ -> matrix multiplication
        
    def loss (self, x, y):
        return torch.mean(torch.square(self.f(x) - y))
    

model = LinearRegresionModel()


optimizer = torch.optim.SGD([model.W, model.b], 0.01)

for epoch in range(1000):
    model.loss(x_train_tensor, y_train_tensor).backward()
    optimizer.step()
    
    optimizer.zero_grad()
    
    
# Print model variables and loss
print("W = %s, b = %s, loss = %s" % (model.W, model.b, model.loss(x_train_tensor, y_train_tensor)))

# Visualize result
plt.plot(x_train_tensor, y_train_tensor, 'o', label='$(x^{(i)},y^{(i)})$')
plt.xlabel('x')
plt.ylabel('y')
x = torch.tensor([[torch.min(x_train_tensor)], [torch.max(x_train_tensor)]])  # x = [[1], [6]]]
plt.plot(x, model.f(x).detach(), label='$\\hat y = f(x) = xW+b$')
plt.legend()
plt.show()