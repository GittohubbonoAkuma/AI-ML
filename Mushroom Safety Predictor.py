import torch
import torch.nn as nn
import torch.optim as optim
#               size,gill size
X=torch.tensor([[0.9,1.0],#small,small,SAFE
                [0.8,0.5],#small,small,SAFE
                [8.5,9.6],#big,big,NOT SAFE
                [9.0,8.7]])#big,big,NOT SAFE
Y=torch.tensor([[0.0],[0.0],[1.0],[1.0]])

class Mushroom(nn.Module):
  def __init__(self):
    super().__init__()
    self.linear=nn.Linear(2,1)
    self.sigmoid=nn.Sigmoid()


  #define the forward pass
  def forward(self,x):
    return self.sigmoid(self.linear(x))

model=Mushroom()

criterion=nn.BCELoss()
optimizer=optim.SGD(model.parameters(),lr=0.01)

for epoch in range(1000):
    prediction=model(X)
    loss=criterion(prediction,Y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch+1)%200==0:
        print(f'Epoch {epoch+1}, Loss {loss.item():.4f}')


data=list(map(float,input("Enter your mushroom's data: ").split()))
test_val=torch.tensor(data)
with torch.no_grad():
  predicted=model(test_val)

print(predicted)
print("Safe to eat" if predicted <0.5 else "Poisonous,don't eat")
    
  
