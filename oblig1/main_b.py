import torch
import matplotlib.pyplot as plt
import csv

#file = open('files/day_length_weight.csv')
#reader = csv.reader(file, delimiter=',')

#header = []
#header = next(reader)


x = []
y =[]
z = []

with open('files/day_length_weight.csv') as dataset:
    data = dataset.readlines()[1:]
    for row in data:
        zs, xs, ys = row.split(',')   #splits on ','
        x.append(float(xs))
        y.append(float(ys))
        z.append(float(zs))
        

x_train = torch.tensor(x).reshape(-1,1)
y_train = torch.tensor(x).reshape(-1,1)
z_train = torch.tensor(x).reshape(-1,1)


class LinearRegressionModel3D:
    def __init__(self):
        self.W1 = torch.tensor([[0.0]], requires_grad=True)
        self.W2 = torch.tensor([[0.0]], requires_grad=True)
        self.b = torch.tensor([[0.0]], requires_grad=True)
        
    #predictor
    def f(self, x1, x2):
        return x1 @ self.W1 + x2 @ self.W2 + self.b
    

    def loss (self, x1, x2, y):
        return torch.nn.functional.mse_loss(self.f(x1,x2), y)


model = LinearRegressionModel3D()


optimizer = torch.optim.SGD([model.W1, model.W2, model.b], 0.0001)

for epoch in range(3000):
    model.loss(x_train, y_train, z_train).backward() #loss gradient
    optimizer.step()
    optimizer.zero_grad()
    
    
# all below in just copy paste just sayin    
# Print model variables and loss
print("W1 = %s, W2 = %s b = %s, loss = %s" % (model.W1, model.W2, model.b, model.loss(x_train, y_train, z_train)))

# Visualize result
fig = plt.figure('Linear regression 3d')
ax = plt.axes(projection='3d', title="Predict days based on length and weight")
# Information for making the plot understandable
ax.set_xlabel('$x_1$')
ax.set_ylabel('$x_2$')
ax.set_zlabel('$y$')
ax.set_xticks([])  # Removes the lines and information from axes
ax.set_yticks([])
ax.set_zticks([])
ax.w_xaxis.line.set_lw(0)
ax.w_yaxis.line.set_lw(0)
ax.w_zaxis.line.set_lw(0)
ax.quiver([0], [0], [0], [torch.max(x_train + 1)], [0],
          [0], arrow_length_ratio=0.05, color='black')
ax.quiver([0], [0], [0], [0], [torch.max(y_train + 1)],
          [0], arrow_length_ratio=0.05, color='black')
ax.quiver([0], [0], [0], [0], [0], [torch.max(z_train + 1)],
          arrow_length_ratio=0, color='black')
# Plot
ax.scatter(x, y, z)
x_tensor = torch.tensor([[torch.min(x_train)], [torch.max(x_train)]])
y_tensor = torch.tensor([[torch.min(y_train)], [torch.max(y_train)]])
ax.plot(x_tensor.flatten(), y_tensor.flatten(), model.f(
    x_tensor, y_tensor).detach().flatten(), label='$f(x)=x1W1+x2W2+b$', color="orange")
# TODO: Fix 3D plane
ax.legend()
plt.show()
    
