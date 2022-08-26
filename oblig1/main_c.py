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

print(x)

x_train = torch.tensor(x).reshape(-1,1)
y_train = torch.tensor(y).reshape(-1,1)

class NonLinearRegressionModel:
    def __init__(self):
        self.W = torch.tensor([[0.0]], requires_grad=True)
        self.b = torch.tensor([[0.0]], requires_grad=True)
        
    def f(self, x):
        return 0
    
    def loss(self, x, y):
        return 0
    