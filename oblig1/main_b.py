import torch
import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv('files/day_length_weight.csv')      

x_train = torch.tensor(data[['length', 'weight']].values)
target = torch.tensor(data['day']).reshape(-1, 1)

class LinearRegressionModel3D:
    def __init__(self):
        self.W1 = torch.tensor([[0.0], [0.0]], requires_grad=True, dtype=torch.double)
        self.b = torch.tensor([[0.0]], requires_grad=True)
        
    #predictor
    def f(self, x1):
        return x1 @ self.W1 + self.b
    

    def loss (self, x1, y):
        return torch.nn.functional.mse_loss(self.f(x1),  y)


model = LinearRegressionModel3D()

loss_record = []


optimizer = torch.optim.Adam([model.W1, model.b], 0.1)

for epoch in range(35000):
    loss = model.loss(x_train, target) #loss gradient
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    
    if (epoch%500 == 0):
        loss_record.append(loss.item())
    
    

print("W = %s, b = %s, loss = %s" % (model.W1, model.b, model.loss(x_train, target)))

ax = plt.axes(projection='3d')
ax.scatter(data['length'], data['weight'], target, c=data['day'], cmap='plasma')


xs = torch.linspace(0, 130, steps = 10)
ys = torch.linspace(0,30, steps = 10)

x,y = torch.meshgrid(xs, ys, indexing='xy')
z = model.f(torch.cat(tuple(torch.dstack([x,y]))).double())
asd = z.reshape(10,10)
ax.plot_surface(x.numpy(), y.numpy(), asd.detach().numpy(), alpha=0.5)
plt.show()
