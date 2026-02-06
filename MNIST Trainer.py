import torch
import torch.nn as nn
import torchvision
import torch.optim as optim
import torchvision.transforms as transforms

#converter to convert PIL images to tensor
transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,),(0.5,))
])

#define the datasets and transform into tensors
train_data=torchvision.datasets.MNIST(root='./data',train=True,download=True,transform=transform)
test_data=torchvision.datasets.MNIST(root='./data',train=False,download=True,transform=transform)

#load the data
#for processing,segmenting into batches
train_loader=torch.utils.data.DataLoader(train_data,batch_size=64,shuffle=True,num_workers=2)
test_loader=torch.utils.data.DataLoader(test_data,batch_size=64,shuffle=False,num_workers=2)

def main():
    class CNN(nn.Module):
        def __init__(self):
            super().__init__()

            self.conv1=nn.Conv2d(1,32,kernel_size=3,padding=1)#Conv2d takes a kernell of size 3*3 moves it over the entire image,
            self.conv2=nn.Conv2d(32,64,kernel_size=3,padding=1)#multiples by some weight

            self.pool=nn.MaxPool2d(2,2)#shrinks the layers(for faster computation) and takes max in each 2*2 matrix

            self.fc1=nn.Linear(64*7*7,128)
            self.fc2=nn.Linear(128,10)

        def forward(self,x):
            x=self.pool(torch.relu(self.conv1(x)))

            x=self.pool(torch.relu(self.conv2(x)))

            x=x.view(x.size(0),-1)
            x=torch.relu(self.fc1(x))
            x=self.fc2(x)

            return x
        

    model=CNN()

    criterion=nn.CrossEntropyLoss()
    optimizer=optim.Adam(model.parameters(),lr=0.001)

    #training
    model.train()#sets the model in trainig mode
    epochs=5
    for epoch in range(epochs):
        running_loss=0

        for images,labels in train_loader:
            output=model(images)
            loss=criterion(output,labels)


            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss+=loss.item()



        print(f'Epoch [{epoch+1}/{epochs}, Loss: {running_loss}]')


    #testing 
    model.eval()#sets the model to testing mode
    correct=0
    total=0

    with torch.no_grad():
        for images,labels in test_loader:
            outputs=model(images)

            _,predicted=torch.max(outputs,1)

            total+=labels.size(0)
            correct+=(predicted==labels).sum().item()
        print(f'Test Accuracy: {100*correct/total:.2f}%')


if __name__=="__main__":
    main()




