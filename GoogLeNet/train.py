import torch
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision import transforms
from model import GoogLeNet
from tqdm import tqdm
from loguru import logger
from utils import part
#如果不处理则会返回PIL图像，因此需要先进行ToTensor和归一化处理
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(0.5,0.5)
])
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
nclass = len(class_names)

train_loader = DataLoader(CIFAR10('./data', train=True, download=True, transform=transform), batch_size=64, shuffle=True)
test_loader = DataLoader(CIFAR10('./data', train=False, download=True, transform=transform), batch_size=64, shuffle=False)


epoch_num = 20

model = GoogLeNet()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
criterion = torch.nn.CrossEntropyLoss()

test_loss = []
train_loss = []
#开始准备训练若干个epoch
for epoch in range(epoch_num):
    logger.info(f'Epoch {epoch}')
    loss_record = []
    success_record = []
    # for data in tqdm(part(train_loader,128)):
    for data in tqdm(train_loader):
        images,labels = data
        optimizer.zero_grad()
        y = torch.nn.functional.one_hot(labels,nclass).to(dtype=torch.float) #这里注意要转化为float型的张量才好求loss
        pred = model(images)
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()
        loss_record.append(loss.detach())
        success_record.append((torch.argmax(pred,dim=1) == labels).to(dtype=int).sum()/labels.shape[0])
    avg_loss = sum(loss_record)/len(loss_record)
    train_loss.append(avg_loss)
    logger.info(f"Train in epoch {epoch} has average train loss:{avg_loss}")
    success_rate = sum(success_record)/len(success_record)
    logger.info(f"Train in epoch {epoch} has success rate:{success_rate}")
    
    #测试模型
    logger.info('testing...')
    with torch.no_grad():
        loss_record = []
        success_record = []
        # for data in tqdm(part(test_loader,32)):
        for data in tqdm(test_loader):
            images,labels = data
            y = torch.nn.functional.one_hot(labels,nclass).to(dtype=torch.float) #训练时的标签是one-hot向量
            pred = model(images)
            loss = criterion(pred, y)
            loss_record.append(loss.detach())
            success_record.append((torch.argmax(pred,dim=1) == labels).to(dtype=int).sum()/labels.shape[0])
        avg_loss = sum(loss_record)/len(loss_record)
        test_loss.append(avg_loss)
        logger.info(f"Test in epoch {epoch} has average test loss:{avg_loss}")
        success_rate = sum(success_record)/len(success_record)
        logger.info(f"Test in epoch {epoch} has success rate:{success_rate}")

    #保存模型
    save_path = f'./GoogLeNet/save/model_ep{epoch}.pth'
    torch.save(model, save_path)
    logger.info(f"Model saved to {save_path}")

#绘制训练损失和测试损失

import matplotlib.pyplot as plt 
plt.figure()
plt.plot(range(epoch_num),train_loss,color='orange',label='train loss')
plt.plot(range(epoch_num),test_loss,color='green',label='test loss')
plt.xlabel('epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()
