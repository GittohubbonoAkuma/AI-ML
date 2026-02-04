import torch
import torch.nn as nn
import torch.optim as optim

X=torch.tensor([[1.0],[2.0],[3.0],[4.0]])
Y=torch.tensor([[2.0],[4.0],[6.0],[8.0]])
#initialize the model
model=nn.Linear(1,1)


#Mean Sqaured Loss
criterion=nn.MSELoss()
optimizer=optim.SGD(model.parameters(),lr=0.01)#learning_rate=0.01

for epoch in range(1000):
  prediction=model(X)
  loss=criterion(prediction,Y)

  optimizer.zero_grad()
  loss.backward()

  optimizer.step()
  if (epoch+1)%100==0:
      print(f'Epoch [{epoch+1/500}],Loss:{loss.item():.4f}')


nums=float(input("Enter a Number"))

test_val=torch.tensor([[nums]])


predicted=model(test_val)
print(f'Predicted {predicted}')
