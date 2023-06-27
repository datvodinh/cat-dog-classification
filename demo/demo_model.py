import torch
import torch.nn as nn
from torchvision import transforms,models
import os

def init_model(model_type:str):
    if model_type=="resnet18":
        resnet = models.resnet18()

        resnet.fc = nn.Sequential(
                    nn.Linear(512, 128),
                    nn.ReLU(inplace=True),
                    nn.Linear(128, 2))
        
        resnet.load_state_dict(torch.load(os.path.join(os.getcwd(),"demo/resnet.pt"),map_location=torch.device('cpu')))

        return resnet
    else:
        cnn_model = nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=32,kernel_size=(3,3)),
            nn.ReLU(),
            # nn.BatchNorm2d(16),
            nn.MaxPool2d(2,2),
            nn.Dropout(p=0.25),
            nn.Conv2d(in_channels=32,out_channels=32,kernel_size=(3,3)),
            nn.ReLU(),
            # nn.BatchNorm2d(32),
            nn.MaxPool2d(2,2),
            nn.Dropout(p=0.25),
            nn.Conv2d(in_channels=32,out_channels=64,kernel_size=(3,3)),
            nn.ReLU(),
            # nn.BatchNorm2d(64),
            nn.MaxPool2d(2,2),
            nn.Dropout(p=0.25),
            nn.Flatten(1),
            nn.Linear(9216,128),
            nn.ReLU(),
            # nn.BatchNorm1d(128),
            nn.Dropout(p=0.4),
            nn.Linear(128,2))
        
        cnn_model.load_state_dict(torch.load(os.path.join(os.getcwd(),"demo/model.pt"),map_location=torch.device('cpu')))

        return cnn_model





train_transform = transforms.Compose([ 
    transforms.Resize((112,112)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def predict(image,model_type:str):
    model = init_model(model_type)
    model.eval()
    data = train_transform(image)
    data = data.reshape(1,3,112,112)
    return model(data).detach().numpy()

