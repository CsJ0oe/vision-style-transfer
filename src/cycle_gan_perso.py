import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import imageio
import os, os.path
import numpy as np
from torch.utils.data import DataLoader
from torch.optim import Adam
import random

#torch.save(model.state_dict(), PATH) # save the model

#model = TheModelClass(*args, **kwargs) # load the model
#model.load_state_dict(torch.load(PATH))
#model.eval()

#nb_training_A = len([name for name in os.listdir('./trainA')])
#nb_training_B = len([name for name in os.listdir('./trainB')])
#ZeroPad2d
MAX_IMAGES = 10
NB_EPOCHS = 10
BATCH_SIZE = 1

def construction_images_tensor(name_dir, max_nb = -1):
    images = []
    liste = os.listdir(name_dir)
    length = len(liste)
    nb = length
    if max_nb != -1 and max_nb < nb: # Ne pas avoir une population écrasante pour les photos, par rapport aux peintures
        nb = max_nb
    for i in range(nb):
        name = liste[random.randint(0, length-1)]
        #images.append(torch.transpose(torch.Tensor(imageio.imread('./trainA/' + name )), -1, 0))
        images.append(imageio.imread(name_dir + name ))
     
    images = np.array(images, dtype = float) / 255
    images = torch.as_tensor(images)
    images = torch.transpose(images, -1,1)
    print("fin du chargement des images de " + name_dir)
    return images



#images = torch.transpose(images, -1, 1) 

#from PIL import Image
#im = Image.open('./trainA/00001.jpg')
#pixels = list(im.getdata())

RES_LAYER_NB = 5

class DataSet:

    def __init__(self, im, lab):
        """Init function should not do any heavy lifting, but
            must initialize how many items are available in this data set.
        """

        #self.ROOT = root
        self.images = im
        self.labels = lab

    def __len__(self):
        """return number of points in our dataset"""

        return len(self.images)

    def __getitem__(self, idx):
        """ Here we have to return the item requested by `idx`
            The PyTorch DataLoader class will use this method to make an iterable for
            our training or validation loop.
        """

        img = self.images[idx]
        label = self.labels[idx]

        return img, label

    def __add__(self, other):
        return ConcatDataset([self, other])


class TransformOtherDimension(nn.Module):

    def __init__(self):
        super(TransformOtherDimension, self).__init__()
        self.downsampling0 = nn.Conv2d(3, 6, 7, stride = 1)
        self.downsampling1 = nn.Conv2d(6, 16, 3, stride = 2, padding = 2)
        self.downsampling2 = nn.Conv2d(16, 32, 3, stride = 2, padding = 2)
        self.padding1 = torch.nn.ZeroPad2d((0,1,0,1))
        self.res_layers = []
        
        for i in range(RES_LAYER_NB):
            tempo = []

            # All the layers in the res layer
            tempo.append(nn.Conv2d(32, 32, 3, padding = 1))
            tempo.append(nn.ReLU())
            tempo.append(nn.Conv2d(32, 32, 3, padding = 1))
            tempo.append(nn.ReLU())
            
            tempo.append(nn.ReLU()) # Used to add the input
            
            self.res_layers.append(tempo)

        self.upsampling1 = nn.ConvTranspose2d(32, 16, 3, stride = 2)
        self.upsampling2 = nn.ConvTranspose2d(16, 8, 3, stride = 2)
        self.padding2 = torch.nn.ZeroPad2d((0,-1,0,-1))
        self.upsampling3 = nn.Conv2d(8, 6, 7, stride = 1)
        self.conv1 = nn.Conv2d(6, 3, 3, stride = 1, padding = 1)
        self.conv2 = nn.Conv2d(3, 3, 3, stride = 1, padding = 1)

        

    def forward(self, x):
        x = self.downsampling0(x)
        #print(x.shape)
        x = self.downsampling1(x)
        #x = self.padding1(x)
        #print(x.shape)
        x = self.downsampling2(x)
        x = self.padding1(x)
        #print(x.shape)

        for i in range(RES_LAYER_NB):
            tempo = torch.clone(x)
            x = self.res_layers[i][0](x)
            x = self.res_layers[i][1](x)
            x = self.res_layers[i][2](x)
            x = self.res_layers[i][3](x)
            
            x += tempo
            x = self.res_layers[i][4](x)
            #print(x.shape)

        x = self.upsampling1(x)
        #print(x.shape)
        x = self.upsampling2(x)
        x = self.padding2(x)
        #print(x.shape)
        x = self.upsampling3(x)
        x = self.conv1(x)
        x = self.conv2(x)
        #print(x.shape)
        return x

class Discriminant(nn.Module):

    def __init__(self):
        super(Discriminant, self).__init__()
        self.padding =  nn.ZeroPad2d((1,1,1,1))
        self.relu = nn.ReLU()
        self.downsampling1 = nn.Conv2d(3, 8, 3, stride = 2)
        self.downsampling2 = nn.Conv2d(8, 16, 3, stride = 2)
        self.downsampling3 = nn.Conv2d(16, 32, 3, stride = 2)
        self.downsampling4 = nn.Conv2d(32, 32, 3, stride = 2)
        self.downsampling5 = nn.Conv2d(32, 1, 3, stride = 2)
        self.sigmoid = nn.Sigmoid()

        

        

    def forward(self, x):
        x = self.downsampling1(x)
        x = self.relu(x)
        x = self.padding(x)
        #print(x.shape)
        x = self.downsampling2(x)
        x = self.relu(x)
        x = self.padding(x)
        #print(x.shape)
        x = self.downsampling3(x)
        x = self.relu(x)
        x = self.padding(x)
        #print(x.shape)
        x = self.downsampling4(x)
        x = self.relu(x)
        x = self.padding(x)
        #print(x.shape)
        x = self.downsampling5(x)
        x = self.sigmoid(x)
        return x



#model.load_state_dict(torch.load(PATH))
def main():
    true_images_A = construction_images_tensor('./trainA/', MAX_IMAGES).float()
    true_images_B = construction_images_tensor('./trainB/', len(true_images_A)).float()
    print(type(true_images_A), type(true_images_B),true_images_A.shape, true_images_B.shape)

    PATH = "./trans_A_to_B" 
    trans_A_to_B = TransformOtherDimension().float()
    if os.path.exists(PATH):
        print("Loading ", PATH)
        trans_A_to_B.load_state_dict(torch.load(PATH))

    PATH = "./discr_B"     
    discr_B = Discriminant().float()
    if os.path.exists(PATH):
        print("Loading ", PATH)
        discr_B.load_state_dict(torch.load(PATH))

    PATH = "./trans_B_to_A" 
    trans_B_to_A = TransformOtherDimension().float()
    if os.path.exists(PATH):
        print("Loading ", PATH)
        trans_B_to_A.load_state_dict(torch.load(PATH))

    PATH = "./discr_A" 
    discr_A = Discriminant().float()
    if os.path.exists(PATH):
        print("Loading ", PATH)
        discr_A.load_state_dict(torch.load(PATH))
        

    opt_trans_A_to_B = Adam(trans_A_to_B.parameters(), lr=0.0001)
    opt_trans_B_to_A = Adam(trans_B_to_A.parameters(), lr=0.0001)
    opt_discr_B = Adam(discr_B.parameters(), lr=0.0001)
    opt_discr_A = Adam(discr_A.parameters(), lr=0.0001)


    #for x in loader_A:
        #print(discr1(x.float()).shape)

    for i in range(NB_EPOCHS):

        print("DEBUT EPOCH ", i)
        print()

        # PARTIE ENTRAINEMENT DISCRIMINATEURS #

        # I: Générer les images fausses
        false_images_B = trans_A_to_B(true_images_A)
        false_images_A = trans_B_to_A(true_images_B)

        if i % 10 == 0:
            pass

        false_labels = torch.as_tensor(np.zeros((len(false_images_A),1,8,8), dtype = float))
        true_labels = torch.as_tensor(np.ones((len(false_images_A),1,8,8), dtype = float))

        ds_A = DataSet(torch.cat((true_images_A,false_images_A), dim=0), torch.cat((true_labels, false_labels), dim=0))
        ds_B = DataSet(torch.cat((true_images_B,false_images_B), dim=0), torch.cat((true_labels, false_labels), dim=0))

        loader_A = DataLoader(ds_A, batch_size = BATCH_SIZE, shuffle = True)
        loader_B = DataLoader(ds_B, batch_size = BATCH_SIZE, shuffle = True)

        total_loss_A = 0.0
        total_loss_B = 0.0
        total_loss_AB = 0.0
        total_loss_BA = 0.0

        k = 0
        
        # II: Apprendre sur les images vraies et fausses
        for img, lab in loader_A:
            img = img.float()
            opt_discr_A.zero_grad()
            predict_A = discr_A(img)
            loss_A = abs(torch.sum(predict_A - lab))
            total_loss_A += loss_A
            loss_A.backward(retain_graph=True)
            opt_discr_A.step()
            k +=1

        total_loss_A /= float(k)

        k = 0
        
        for img, lab in loader_B:
            img = img.float()
            opt_discr_B.zero_grad()
            predict_B = discr_B(img)
            loss_B = abs(torch.sum(predict_B - lab))
            total_loss_B += loss_B
            loss_B.backward(retain_graph=True)
            opt_discr_B.step()
            k += 1

        total_loss_B /= float(k)

        k = 0
        
        # III: Apprendre la génération
        ds = DataSet(true_images_A, true_images_B)
        loader = DataLoader(ds, batch_size = BATCH_SIZE, shuffle = True)

        for img_A, img_B in loader:
            img_A = img_A.float()
            img_B = img_B.float()
            opt_trans_B_to_A.zero_grad()
            opt_trans_A_to_B.zero_grad()
            length = len(true_images_A)
            false_images_B = trans_A_to_B(img_A)
            false_images_A = trans_B_to_A(img_B)

            discr_images_A = discr_A(false_images_A)
            discr_images_B = discr_B(false_images_B)
            validity_loss_AB = abs(torch.sum(discr_images_B - true_labels)) / length # On veut que le discriminateur se trompe
            validity_loss_BA = abs(torch.sum(discr_images_A - true_labels)) / length
            
            recon_images_A = trans_B_to_A(false_images_B)
            recon_images_B = trans_A_to_B(false_images_A)
            reconstruction_loss_BA = abs(torch.sum(recon_images_A - true_images_A)) / length # Et que la reconstruction des images se passe bien
            reconstruction_loss_AB = abs(torch.sum(recon_images_B - true_images_B)) / length

            ident_A = trans_B_to_A(true_images_A)
            ident_B = trans_A_to_B(true_images_B)
            identity_loss_AB = abs(torch.sum(ident_B - true_images_B)) / length # Et qu'une image avec son propre style ne change pas trop
            identity_loss_BA = abs(torch.sum(ident_A - true_images_A)) / length

            full_loss_AB = (validity_loss_AB * 20 + reconstruction_loss_AB + identity_loss_AB ) / 22
            full_loss_BA = (validity_loss_BA * 20 + reconstruction_loss_BA + identity_loss_BA ) / 22

            total_loss_AB += full_loss_AB
            total_loss_BA += full_loss_BA

            full_loss_AB.backward(retain_graph=True)
            full_loss_BA.backward()

            opt_trans_B_to_A.step()
            opt_trans_A_to_B.step()

            k += 1

        total_loss_AB /= float(k)
        total_loss_BA /= float(k)

        print("FULL LOSS A", total_loss_A)
        print("FULL LOSS B", total_loss_B)
        print("FULL LOSS A->B:", total_loss_AB)
        print("FULL LOSS B->A:", total_loss_BA)
        print("#########################################")

    torch.save(trans_B_to_A.state_dict(), "./trans_B_to_A") # save the model
    torch.save(trans_A_to_B.state_dict(), "./trans_A_to_B") # save the model
    torch.save(discr_A.state_dict(), "./discr_A") # save the model
    torch.save(discr_B.state_dict(), "./discr_B") # save the model

main()
