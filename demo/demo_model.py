import torch
import torch.nn as nn
from torchvision import transforms,models
import streamlit as st
def init_model():
    resnet = models.resnet18()

    resnet.fc = nn.Sequential(
                nn.Linear(512, 128),
                nn.ReLU(inplace=True),
                nn.Linear(128, 2))

    resnet.load_state_dict(torch.load("/model/resnet.pt",map_location=torch.device('cpu')))
    return resnet

train_transform = transforms.Compose([ 
    transforms.Resize((112,112)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def predict(image):
    model = init_model()
    model.eval()
    data = train_transform(image)
    data = data.reshape(1,3,112,112)
    return model(data).detach().numpy()

