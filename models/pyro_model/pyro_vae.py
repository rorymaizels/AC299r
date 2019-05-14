import pt_helper # helper functions for loading data

import torch
import torch.nn as nn
import numpy as np
import pyro
from torch.utils.data import DataLoader
import pyro.distributions as dist

# set up
pyro.enable_validation(True)
pyro.set_rng_seed(0)


class Decoder(nn.Module):
    """
    takes latent variables z, passes through two hidden layers and returns reconstructed x.
    """
    def __init__(self, alph_size,seq_len, z_dim=30, hidden_architecture=[100,500]):
        super(Decoder, self).__init__()
        # setup the two linear transformations used
        self.hidden1 = nn.Linear(z_dim,hidden_architecture[0])
        self.hidden2 = nn.Linear(hidden_architecture[0],hidden_architecture[1])
        self.final = nn.Linear(hidden_architecture[1],(alph_size*seq_len))
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, z):
        # define the forward computation on the latent z
        hidden1 = self.relu(self.hidden1(z))
        hidden2 = self.sigmoid(self.hidden2(hidden1))
        output = self.sigmoid(self.final(hidden2))
        return output


class Encoder(nn.Module):
    """
    Takes in data, returns mu and sigma for variational approximation of latent variable.
    """
    def __init__(self,alph_size,seq_len, z_dim=30, hidden_architecture=[1500,1500]):
        super(Encoder, self).__init__()
        self.hidden1 = nn.Linear((alph_size*seq_len),hidden_architecture[0])
        self.hidden2 = nn.Linear(hidden_architecture[0],hidden_architecture[1])
        self.final1 = nn.Linear(hidden_architecture[1],z_dim)
        self.final2 = nn.Linear(hidden_architecture[1],z_dim)
        self.relu = nn.ReLU()
        self.alph_size = alph_size
        self.seq_len = seq_len

    def forward(self, x):
        x = x.reshape(-1,self.seq_len*self.alph_size)
        hidden1 = self.relu(self.hidden1(x))
        hidden2 = self.relu(self.hidden2(hidden1))
        z_loc = self.final1(hidden2)
        z_scale = torch.exp(self.final2(hidden2))
        return z_loc, z_scale


class VAE(nn.Module):
    def __init__(self, alph_size, seq_len, z_dim=30, encoder_architecture=[1500, 1500],
                 decoder_architecture=[100, 500], use_cuda=False):
        """
        Variational Autoencoder that defines the pyro structure of model and guide

        :param alph_size: size of alphabet
        :param seq_len: length of sequence
        :param z_dim: dimensions of latent space
        :param encoder_architecture: nodes per layer for encoder
        :param decoder_architecture: nodes per layer for decoder
        :param use_cuda: GPU command
        """
        super(VAE, self).__init__()
        # create the encoder and decoder networks
        # call classes shared by nn.Module
        self.encoder = Encoder(alph_size,seq_len,z_dim, encoder_architecture)
        self.decoder = Decoder(alph_size,seq_len,z_dim, decoder_architecture)
        if use_cuda:
            self.cuda()
        # parameters required in functions
        self.use_cuda = use_cuda
        self.z_dim = z_dim
        self.alph_size = alph_size
        self.seq_len = seq_len

    # define the model for conditional distribution p(x|z)p(z)
    def model(self, x):
        pyro.module('decoder', self.decoder) # adds decoder as a pyro module
        with pyro.plate('data', x.shape[0]):
            z_loc = x.new_zeros(torch.Size((x.shape[0], self.z_dim)))
            z_scale = x.new_ones(torch.Size((x.shape[0], self.z_dim)))
            z = pyro.sample('latent', dist.Normal(z_loc, z_scale).to_event(1))
            output = self.decoder.forward(z)
            # score against actual images
            pyro.sample('obs', dist.Bernoulli(output).to_event(1),
                        obs=x.reshape(-1, self.alph_size*self.seq_len))

    # define the guide (i.e. variational distribution) q(z|x)
    def guide(self, x):
        pyro.module('encoder', self.encoder)
        with pyro.plate('data', x.shape[0]):
            z_loc, z_scale = self.encoder.forward(x)
            # sample the latent code z
            pyro.sample("latent", dist.Normal(z_loc, z_scale).to_event(1))

    def reconstruct_output(self, x):
        z_loc, z_scale = self.encoder(x)
        z = dist.Normal(z_loc, z_scale).sample()
        # decode the image (note not sampled; this is non-bayesian)
        output = self.decoder(z)
        return output


def loader_function(data, bs, nw, pm):
    """
    :param data: dataset specification for datahelper
    :param bs: batch size
    :param nw: number of workers for cuda
    :param pm: pin memory for GPU
    :return: a pytorch dataloader, alphabet and sequence size from datahelper
    """
    datahelper = pt_helper.DataHelper(dataset=data,calc_weights=True)
    x_train = datahelper.x_train.astype(np.float32)
    alph_size = datahelper.alphabet_size
    seq_len = datahelper.seq_len
    data_loader = DataLoader(x_train,batch_size=bs,
                             shuffle=True, num_workers=nw, pin_memory=pm)
    return data_loader, alph_size, seq_len


def train(svi, loader, use_cuda=False):
    """
    per epoch training function.

    :param svi: pyro svi module
    :param loader: data loader from loader_function
    :param use_cuda: GPU command
    :return: loss for that epoch
    """
    epoch_loss = 0.
    for x in loader:
        if use_cuda:
            x = x.cuda()
        epoch_loss += svi.step(x)

    normalizer_train = len(loader.dataset)
    total_epoch_loss_train = epoch_loss / normalizer_train
    return total_epoch_loss_train



