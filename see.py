import torch
from nn import AutoEncoder, DatasetFolder
from torchinfo import summary

feedback_bits = 384
model = AutoEncoder(feedback_bits)
summary(model, input_size=(1, 128,126,2),depth=5,verbose=1)
