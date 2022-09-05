
# Lag en modell som predikerer tilsvarende NOT-operatoren.
# Visulaiser resultatet etter optimilisering av modellen.
import torch
import matplotlib.pyplot as plt

x_train = torch.tensor([[1.0], [0.0]]).reshape(-1,1)
y_train = torch.tensor([[0.0], [1.0]]).reshape(-1,1)

class NOT:
    def __init__(self):
        self.W = torch.tensor([[0.0]], requires_grad=True)
        self.b = torch.tensor([[0.0]], requires_grad=True)
        
        
    def f(self, x):
        return torch.sigmoid(self.logits(x))
    
    def logits(self, x):
        return x @ self.W + self.b
    
    def loss(self, x, y):
        return torch.nn.functional.binary_cross_entropy_with_logits(self.logits(x), y)
    
    

model_not = NOT()

optmizer = torch.optim.Adam([model_not.b, model_not.W], 0.1)

for epoch in range(100000):
    model_not.loss(x_train, y_train).backward()
    optmizer.step()
    optmizer.zero_grad()
    
print("W=%s, b=%s, loss=%s" %(model_not.W, model_not.b, model_not.loss(x_train, y_train)))

plt.plot(x_train, y_train, 'o', label='$(x^{(i)},y^{(i)})$')
plt.xlabel('x')
plt.ylabel('y')
x = torch.arange(0.0, 1.0, 0.01).reshape(-1, 1)
plt.plot(x, model_not.f(x).detach(), label='$\\hat y = f(x) = \sigma(xW + b)$')
plt.legend()
plt.show()

