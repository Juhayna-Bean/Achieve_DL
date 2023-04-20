import torch
import torch.nn as nn
import torch.nn.functional as F

#1x1卷积
def conv1x1(in_channels, out_channels):
    return nn.Sequential(nn.Conv2d(in_channels, out_channels, 1),nn.LeakyReLU())

#3x3卷积
def conv3x3(in_channels, out_channels, stride=1, padding=1):
    return nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, stride, padding),nn.LeakyReLU())

#Inception块
class InceptionBlock(nn.Module):
    def __init__(self, in_channels, out_channels, config) -> None:
        super().__init__()
        self.paths = nn.ModuleList()
        self.skip_connect = config['skip_connect']
        if config['use_1x1_3x3']:
            self.paths.append(nn.Sequential(
                conv1x1(in_channels,out_channels),
                conv3x3(out_channels,out_channels)
            ))
        if config['use_1x1_3x3_3x3']:
            self.paths.append(nn.Sequential(
                conv1x1(in_channels,out_channels),
                conv3x3(out_channels,out_channels,),
                conv3x3(out_channels,out_channels,)
            ))
        if config['use_1x1_1x7_7x1']:
            self.paths.append(nn.Sequential(
                conv1x1(in_channels,out_channels),
                nn.Conv2d(out_channels,out_channels,(1,7),padding=(0,3)),
                nn.Conv2d(out_channels,out_channels,(7,1),padding=(3,0))
            ))
        if config['use_pool_1x1']:
            self.paths.append(nn.Sequential(
                nn.MaxPool2d(3,stride=1,padding=1),
                conv1x1(in_channels, out_channels)
            ))
        if config['use_1x1']:
            self.paths.append(conv1x1(in_channels, out_channels))

    def forward(self, x):
        y = [path(x) for path in self.paths]
        
        y = torch.cat(y,dim=1) #在channel这个维度将其连接起来
        if self.skip_connect:
            y = y + x.repeat(1,y.shape[1]//x.shape[1],1,1) #注意由于通道数发生改变，要进行复制拓展
        return F.leaky_relu(y)

class GoogLeNet(nn.Module):
    def __init__(self, num_classes = 10, **kwargs) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.conv3x3 = nn.Conv2d(3, 1, 3, padding=1)

        #配置inception块的内容
        self.config = {
            'use_1x1_3x3':True,
            'use_1x1_3x3_3x3':True,
            'use_1x1_1x7_7x1':True,
            'use_pool_1x1':True,
            'use_1x1':True,
            'skip_connect':True, #跨越连接
            'dropout':False,
            'batch_norm':True
        }
        self.config.update(kwargs)
        
        self.stem = nn.Sequential(
            conv3x3(3,16,stride=2),
            conv3x3(16,16,stride=1),
            conv3x3(16,32,stride=1),
            nn.MaxPool2d(3,stride=1),
            conv1x1(32,64),
            conv3x3(64,64),
            conv3x3(64,128),
            nn.MaxPool2d(3,stride=2)
        )

        self.inception1 = InceptionBlock(128, 256, self.config)
        self.avg_pool = nn.AvgPool2d(6,6)
        if self.config['dropout']:
            self.dropout = nn.Dropout(0.5)
        if self.config['batch_norm']:
            self.batch_norm = nn.BatchNorm1d(1280)

        self.lin = nn.Linear(1280,10)

    def forward(self, x):

        y = self.stem(x)
        y = self.inception1(y)
        y = self.avg_pool(y)
        y = torch.squeeze(y)
        if self.config['dropout']:
            y = self.dropout(y)
        if self.config['batch_norm']:
            y = self.batch_norm(y)
        y = self.lin(y)
        return y


if __name__ == '__main__':
    from utils import get_demo_data,get_model_size
    images, labels = get_demo_data()
    print('images:',images.shape)
    print('labels',labels.shape)
    model = GoogLeNet(use_1x1_3x3=True,
            use_1x1_3x3_3x3=True,
            use_1x1_1x7_7x1=True,
            use_pool_1x1=True,
            use_1x1=True,
            drop_out=True,
            batch_norm=False 
            )
    print("model size:", get_model_size(model))
    with torch.no_grad():
        y_pred = model(images)
    print('y_pred:',y_pred.shape)
   
