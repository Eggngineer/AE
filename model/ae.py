# Pytorch Reference
import torch
import torch.nn as nn
import torch.nn.functional as F

# class Encoder
class Encoder(nn.Module):
  def __init__(self, input_dim, hidden_dim, latent_dim):
    super(Encoder, self).__init__()
    self.fc = nn.Linear(input_dim, hidden_dim)
    self.fc_mu = nn.Linear(hidden_dim, latent_dim)

  def forward(self, x):
    h = torch.relu(self.fc(x))
    mu = self.fc_mu(h)

    return mu, None


# class : Decoer
class Decoder(nn.Module):
  def __init__(self, input_dim, hidden_dim, latent_dim):
    super(Decoder, self).__init__()
    self.fc = nn.Linear(latent_dim, hidden_dim)
    self.fc_output = nn.Linear(hidden_dim, input_dim)

  def forward(self, z):
    h = torch.relu(self.fc(z))
    output = torch.sigmoid(self.fc_output(h))
    return output 


# class : AE
class AE(nn.Module):
  def __init__(self, input_dim, hidden_dim, latent_dim):
    super(AE, self).__init__()
    self.encoder = Encoder(input_dim, hidden_dim, latent_dim)
    self.decoder = Decoder(input_dim, hidden_dim, latent_dim)

  def forward(self, x):
    mu, _ = self.encoder(x)
    x_decoded = self.decoder(mu)
    return x_decoded, mu