import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import torch.optim as optim 
import matplotlib.pyplot as plt 
import numpy as np 
import os
from tqdm import tqdm


class random_fourier_features(nn.Module):
    def __init__(self,input_size,output_size,std):
        super().__init__()
        self.random = torch.randn(input_size,output_size)*std

    def forward(self,x):
        return torch.sin(torch.matmul(x,self.random))


class autoencoder(nn.Module):
    def __init__(self,input_size=10,hidden_size=100,latent_size=10,std=10):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(input_size,hidden_size),nn.ReLU(),nn.Linear(hidden_size,hidden_size),nn.ReLU(),nn.Linear(hidden_size,latent_size))
        self.decoder = nn.Sequential(nn.Linear(latent_size,hidden_size),nn.ReLU(),nn.Linear(hidden_size,hidden_size),nn.ReLU(),nn.Linear(hidden_size,input_size))
        self.random_fourier_features = random_fourier_features(2,10,std)
    def forward(self,x):
        input_data = self.random_fourier_features(x)
        latent_rep = self.encoder(input_data)
        output_data = self.decoder(latent_rep)

        return input_data,output_data

if __name__ == "__main__":
    torch.manual_seed(0)
    #hyper 
    n = 50
    x = torch.linspace(0,1,n)
    y = torch.linspace(0,1,n)
    decorrelation_constant = 10
    full_screen = True
    #__init__
    grid_x, grid_y = torch.meshgrid(x, y)
    cooridantes = torch.cat((grid_x.reshape(-1,1),grid_y.reshape(-1,1)),dim=1).reshape(n*n,2)
    grid_known_unknown = torch.bernoulli(torch.ones(n,n)*0.01)

    fig,axis = plt.subplots(1,3)
    
    autoencoder_model = autoencoder(std=decorrelation_constant)
    optimizer = optim.Adam(autoencoder_model.parameters(),lr=0.01)
    
    if(full_screen):#full screen
        mng = plt.get_current_fig_manager()
        mng.full_screen_toggle()

    #train loop 

    for i in tqdm(range(1000),ascii=True):
        optimizer.zero_grad()
        
        known_points = torch.cat([cooridantes[i].view(1,2) for i in range(len(grid_known_unknown.view(-1))) if grid_known_unknown.view(-1)[i]])
        
        input_data, output_data = autoencoder_model(known_points)
        loss = nn.MSELoss()(input_data,output_data)
        loss.backward()

        optimizer.step()
        
        with torch.no_grad():
            input_data, output_data = autoencoder_model(cooridantes)
            believe_state_grid = torch.sum(torch.abs(input_data- output_data),dim=1) 
            reconstruct = believe_state_grid < 0.5


            axis[0].cla()
            axis[1].cla()
            axis[2].cla()
            axis[0].imshow(grid_known_unknown)
            axis[1].imshow(believe_state_grid.view(n,n))
            axis[2].imshow(reconstruct.view(n,n))

            axis[0].title.set_text("visible points / visited states")
            axis[1].title.set_text("error of autoencoder each state/point")
            axis[2].title.set_text("reconstructed visible points (from error)")
            plt.pause(0.01)
            
