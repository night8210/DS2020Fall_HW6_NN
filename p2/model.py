import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.cnn = nn.Sequential(
                nn.Conv2d(3,16,kernel_size=5,stride=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2),
                
                #TODO:
            
                nn.Dropout2d()
            )
        self.avg_pool = nn.AdaptiveAvgPool2d((4,4))
        self.fc = nn.Linear(16*4*4, 2)
        
    def forward(self,x):
        bsize = x.size(0)
        
        x = self.cnn(x)
        x = self.avg_pool(x)
        x = self.fc(x.view(bsize,-1))
        return x
    
