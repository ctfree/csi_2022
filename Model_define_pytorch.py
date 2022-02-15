#!/usr/bin/env python3
"""An Implement of an autoencoder with pytorch.
This is the template code for 2020 NIAC https://naic.pcl.ac.cn/.
The code is based on the sample code with tensorflow for 2020 NIAC and it can only run with GPUS.
Note:
    1.This file is used for designing the structure of encoder and decoder.
    2.The neural network structure in this model file is CsiNet, more details about CsiNet can be found in [1].
[1] C. Wen, W. Shih and S. Jin, "Deep Learning for Massive MIMO CSI Feedback", in IEEE Wireless Communications Letters, vol. 7, no. 5, pp. 748-751, Oct. 2018, doi: 10.1109/LWC.2018.2818160.
    3.The output of the encoder must be the bitstream.
"""
import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


# This part implement the quantization and dequantization operations.
# The output of the encoder must be the bitstream.
def Num2Bit(Num, B):
    Num_ = Num.type(torch.uint8)

    def integer2bit(integer, num_bits=B * 2):
        dtype = integer.type()
        exponent_bits = -torch.arange(-(num_bits - 1), 1).type(dtype)
        exponent_bits = exponent_bits.repeat(integer.shape + (1,))
        out = integer.unsqueeze(-1) // 2 ** exponent_bits
        return (out - (out % 1)) % 2

    bit = integer2bit(Num_)
    bit = (bit[:, :, B:]).reshape(-1, Num_.shape[1] * B)
    return bit.type(torch.float32)


def Bit2Num(Bit, B):
    Bit_ = Bit.type(torch.float32)
    Bit_ = torch.reshape(Bit_, [-1, int(Bit_.shape[1] / B), B])
    num = torch.zeros(Bit_[:, :, 1].shape).cuda()
    for i in range(B):
        num = num + Bit_[:, :, i] * 2 ** (B - 1 - i)
    return num


class Quantization(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, B):
        ctx.constant = B
        step = 2 ** B
        out = torch.round(x * step - 0.5)
        out = Num2Bit(out, B)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        # return as many input gradients as there were arguments.
        # Gradients of constant arguments to forward must be None.
        # Gradient of a number is the sum of its B bits.
        b, _ = grad_output.shape
        grad_num = torch.sum(grad_output.reshape(b, -1, ctx.constant), dim=2) / ctx.constant
        return grad_num, None


class Dequantization(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, B):
        ctx.constant = B
        step = 2 ** B
        out = Bit2Num(x, B)
        out = (out + 0.5) / step
        return out

    @staticmethod
    def backward(ctx, grad_output):
        # return as many input gradients as there were arguments.
        # Gradients of non-Tensor arguments to forward must be None.
        # repeat the gradient of a Num for B time.
        b, c = grad_output.shape
        grad_output = grad_output.unsqueeze(2) / ctx.constant
        grad_bit = grad_output.expand(b, c, ctx.constant)
        return torch.reshape(grad_bit, (-1, c * ctx.constant)), None


class QuantizationLayer(nn.Module):

    def __init__(self, B):
        super(QuantizationLayer, self).__init__()
        self.B = B

    def forward(self, x):
        out = Quantization.apply(x, self.B)
        return out


class DequantizationLayer(nn.Module):

    def __init__(self, B):
        super(DequantizationLayer, self).__init__()
        self.B = B

    def forward(self, x):
        out = Dequantization.apply(x, self.B)
        return out



class MLP(nn.Module):
    def __init__(self, input_dim, dims, activation=None):
        super().__init__()
        layers = []
        for dim in dims[:-1]:
            layers.append(nn.Linear(input_dim, dim))
            layers.append(_get_activation_layer(activation))
            input_dim = dim
        layers.append(nn.Linear(input_dim, dims[-1]))
        self._mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self._mlp(x)

def _get_activation_layer(activation):
    if activation == "relu":
        return nn.ReLU()
    elif activation == "gelu":
        return nn.GELU()
    elif activation == 'tanh':
        return nn.Tanh()
    else:
        return getattr(nn, activation)()




class TransformerEncoder(torch.nn.TransformerEncoder):
    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__(encoder_layer, num_layers, norm)
        self._reset_parameters()
    def forward(self, src, mask=None, src_key_padding_mask=None, return_hidden=False):
        output = src
        outputs = []
        for mod in self.layers:
            output = mod(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask)
            if return_hidden:
                outputs.append(output.detach())

        if self.norm is not None:
            output = self.norm(output)
            if return_hidden:
                outputs[-1] = output.detach()
        if return_hidden:
            self.enc_hidden = torch.stack(outputs, 1)
        return output
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)


class TransformerEncoderLayer(nn.TransformerEncoderLayer):
    pass

class TransformerDecoderLayer(nn.TransformerDecoderLayer):
    pass


class ALBert(TransformerEncoder):
    def __init__(self, encoder_layer, num_layers, norm=None):
        super(TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([encoder_layer for i in range(num_layers)])
        self.num_layers = num_layers
        self.norm = norm


class Module(nn.Module):
    batch_first = False
    def __init__(self, cfg, **kwargs):
        if 'encoder' in kwargs:
            encoder = kwargs.pop('encoder')
        super(Module, self).__init__(**kwargs)
        self.cfg = deepcopy(cfg)
        self.create_layers()
        if not self.cfg.no_init:
            self.apply(self.init_weights)
        if hasattr(self.cfg, 'min_v'):
            self.min_v = torch.nn.Parameter(torch.FloatTensor(np.array(self.cfg.min_v)), requires_grad=False)
            self.max_v = torch.nn.Parameter(torch.FloatTensor(np.array(self.cfg.max_v)), requires_grad=False)
        self.mean_v = torch.nn.Parameter(torch.FloatTensor(np.array(self.cfg.mean_v)), requires_grad=False)
        self.std_v = torch.nn.Parameter(torch.FloatTensor(np.array(self.cfg.std_v)), requires_grad=False)

    def init_weights_bak(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(
                mean=0.0, std=self.cfg.initializer_range)
        elif isinstance(module, torch.nn.MultiheadAttention):
            self._reset_parameters(module)

        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def init_weights(self, module):
        """ Initialize the weights.
        """
        # print(module)
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(
                mean=0.0, std=self.cfg.initializer_range)
        elif isinstance(module, nn.Parameter):
            module.weight.data.normal_(
                mean=0.0, std=self.cfg.initializer_range)

        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def _reset_parameters(self, module):
        if module._qkv_same_embed_dim:
            nn.init.normal_(module.in_proj_weight, mean=0.0, std=self.cfg.initializer_range)
        else:
            nn.init.normal_(module.q_proj_weight, mean=0.0, std=self.cfg.initializer_range)
            nn.init.normal_(module.k_proj_weight, mean=0.0, std=self.cfg.initializer_range)
            nn.init.normal_(module.v_proj_weight, mean=0.0, std=self.cfg.initializer_range)

        if module.in_proj_bias is not None:
            nn.init.zeros_(module.in_proj_bias)
            nn.init.zeros_(module.out_proj.bias)
        if module.bias_k is not None:
            nn.init.zeros_(module.bias_k)
        if module.bias_v is not None:
            nn.init.zeros_(module.bias_v)


    def create_transformer_encoder(self, d_model, n_layer, n_head, d_ff):
        if self.cfg.ffd == 'glu':
            encoder_layer = GluEncoderLayer(self.cfg)
        else:
            encoder_layer = TransformerEncoderLayer(d_model=d_model, nhead=n_head, dim_feedforward=d_ff,
                                                   dropout=self.cfg.dropout, activation=self.cfg.activation)
        if self.cfg.share_weights:
            return ALBert(encoder_layer, num_layers=n_layer, norm=None)
        else:
            return TransformerEncoder(encoder_layer, num_layers=n_layer, norm=None)

    def create_layers(self):
        pass

    def forward(self, **kwargs):
        pass


class Encoder(Module):
    def __init__(self, cfg, **kwargs):
        cfg = deepcopy(cfg)
        if cfg.shift_dim:
            dim1 = cfg.dim1
            cfg.dim1 = cfg.dim2
            cfg.dim2 = dim1
        super().__init__(cfg, **kwargs)

    def create_layers(self):
        self.encoder = self.create_transformer_encoder(self.cfg.d_emodel, self.cfg.n_e_layer, self.cfg.n_ehead, self.cfg.d_eff)
        if self.cfg.use_bn:
            self.encoder_norm = torch.nn.BatchNorm1d(self.cfg.d_emodel, eps=1e-05, momentum=self.cfg.momentum, affine=True, track_running_stats=False)
            logger.info('use BN')
        elif self.cfg.use_in:
            logger.info('use IN')
            self.encoder_norm = torch.nn.InstanceNorm1d(self.cfg.d_emodel, eps =1e-05, momentum=self.cfg.momentum, affine=False, track_running_stats=False)

        self.enc_token = torch.nn.Parameter(torch.randn((1, 1, self.cfg.d_emodel)))
        self.quantize_layer = self.create_quantization_layer()
        self.create_pos_emb()
        self.create_ffd()

    def create_quantization_layer(self):
        if self.cfg.mixq_factor>0 and self.cfg.n_q_bit>1:
            quantize_layer = MixQuantizationLayer(self.cfg, self.cfg.enc_dim)
        else:
            quantize_layer = QuantizationLayer(self.cfg.n_q_bit)
        return quantize_layer

    def create_pos_emb(self):
        self.pos_emb = nn.Parameter(self.cfg.initializer_range*torch.randn(self.cfg.dim1+1, self.cfg.d_emodel))

    def create_ffd(self):
        self.input_ffd_layer = MLP(self.cfg.dim2 * self.cfg.dim3, [self.cfg.d_emodel], activation=self.cfg.activation)
        self.output_ffd_layer = MLP(self.cfg.d_emodel, [self.cfg.enc_dim], activation=self.cfg.activation)

    def pre_quantize(self, x):
        x = x*(2**self.cfg.n_q_bit) - 0.5
        return x

    def get_encode_input(self, csi):
        csi = self.input_ffd_layer(csi.reshape(-1, self.cfg.dim1, self.cfg.dim2 * self.cfg.dim3))
        inputs = torch.cat([self.enc_token.repeat(csi.shape[0], 1, 1), csi], 1)
        inputs += self.pos_emb
        return inputs

    def get_encode_output(self, enc):
        enc = enc[:, 0]
        enc = self.output_ffd_layer(enc)
        enc = F.sigmoid(enc)
        return enc

    def encode(self, csi, return_hidden=False, csi_power=None, **kwargs):
        inputs = self.get_encode_input(csi)
        if not self.batch_first:
            enc = self.encoder(torch.transpose(inputs, 0, 1), return_hidden=return_hidden)
            enc = torch.transpose(enc, 0, 1)
        else:
            enc = self.encoder(inputs, return_hidden=return_hidden)
        if self.cfg.use_bn or self.cfg.use_in:
            enc = self.encoder_norm(enc.transpose(1, 2)).transpose(1,2)
        enc = self.get_encode_output(enc)
        return enc

    def quantize(self, enc):
        enc = self.quantize_layer(enc)
        return enc

    def forward(self, csi, return_hidden=False, **kwargs):
        csi = (csi-self.mean_v)/self.std_v
        power_csi = None
        if self.cfg.shift_dim:
            csi = csi.transpose(1, 2)
        enc = self.encode(csi, return_hidden=return_hidden, csi_power=power_csi, **kwargs)
        enc_preq = self.pre_quantize(enc)
        if not self.training or self.cfg.train_quantize:
            enc = self.quantize(enc_preq)
        else:
            enc = enc_preq
        if return_hidden:
            self.enc_enc = enc.detach()
        if not self.cfg.is_test:
            return enc, enc_preq
        else:
            return enc


class Decoder(Module):
    def __init__(self, cfg, **kwargs):
        cfg = deepcopy(cfg)
        if cfg.shift_dim:
            dim1 = cfg.dim1
            cfg.dim1 = cfg.dim2
            cfg.dim2 = dim1
        super().__init__(cfg, **kwargs)
    def create_layers(self):
        self.decoder = self.create_transformer_encoder(self.cfg.d_dmodel, self.cfg.n_d_layer, self.cfg.n_dhead, self.cfg.d_dff)
        if self.cfg.use_bn:
            self.decoder_norm = torch.nn.BatchNorm1d(self.cfg.d_dmodel, eps=1e-05, momentum=self.cfg.momentum, affine=True, track_running_stats=False)
            logger.info('use BN')
        elif self.cfg.use_in:
            logger.info('use IN')
            self.decoder_norm = torch.nn.InstanceNorm1d(self.cfg.d_dmodel, eps =1e-05, momentum=self.cfg.momentum, affine=False, track_running_stats=False)
        self.dec_tokens = torch.nn.Parameter(torch.randn((1, self.cfg.dim1, self.cfg.d_dmodel)))
        self.dequantize_layer = self.create_quantization_layer()
        self.create_pos_emb()
        self.create_ffd()

    def create_quantization_layer(self):
        if self.cfg.mixq_factor>0 and self.cfg.n_q_bit>1:
            quantize_layer = MixDequantizationLayer(self.cfg, self.cfg.enc_dim)
        else:
            quantize_layer = DequantizationLayer(self.cfg.n_q_bit)
        return quantize_layer

    def create_pos_emb(self):
        self.pos_emb = nn.Parameter(self.cfg.initializer_range*torch.randn(self.cfg.dim1+1, self.cfg.d_dmodel))

    def create_ffd(self):
        self.input_ffd_layer = MLP(self.cfg.enc_dim, [self.cfg.d_dmodel], activation=self.cfg.activation)
        self.output_ffd_layer = MLP(self.cfg.d_dmodel, [self.cfg.dim2 * self.cfg.dim3], activation=self.cfg.activation)

    def get_decode_input(self, enc):
        enc = self.input_ffd_layer(enc)
        inputs = torch.cat([enc[:, None], self.dec_tokens.repeat(enc.shape[0], 1, 1)], 1)
        inputs += self.pos_emb
        return inputs

    def get_decode_output(self, dec):
        dec = dec[:, 1:]
        dec = self.output_ffd_layer(dec)
        dec = F.sigmoid(dec)
        return dec


    def decode(self, enc, return_hidden=False, **kwargs):
        inputs = self.get_decode_input(enc)
        if not self.batch_first:
            dec = self.decoder(torch.transpose(inputs, 0, 1), return_hidden=return_hidden)
            dec = torch.transpose(dec, 0, 1)
        else:
            dec = self.decoder(inputs, return_hidden=return_hidden)
        if self.cfg.use_bn or self.cfg.use_in:
            dec = self.decoder_norm(dec.transpose(1, 2)).transpose(1,2)

        dec = self.get_decode_output(dec)
        dec = dec.reshape(-1, self.cfg.dim1, self.cfg.dim2, self.cfg.dim3)
        return dec

    def post_dequantize(self, x):
        x = (x+0.5)/(2**self.cfg.n_q_bit)
        return x

    def dequantize(self, enc):
        enc = self.dequantize_layer(enc)
        return enc

    def forward(self, enc, return_hidden=False, **kwargs):
        if not self.training or self.cfg.train_quantize:
        #if 1==1:
            if not self.cfg.no_deq:
                enc = self.dequantize(enc)
        if not self.cfg.no_deq:
            enc = self.post_dequantize(enc)
        dec = self.decode(enc, return_hidden=return_hidden, **kwargs)
        if self.cfg.use_mm:
            dec = dec*(self.max_v-self.min_v) + self.min_v
        if self.cfg.shift_dim:
            dec = dec.transpose(1, 2)
        return dec

class EncoderPlus(Encoder):
    def create_ffd(self):
        self.input_ffd_layer = MLP(self.cfg.dim2 * self.cfg.dim3, [self.cfg.d_emodel], activation=self.cfg.activation)
        self.output_ffd_layer = MLP(self.cfg.d_emodel*self.cfg.dim1, [self.cfg.enc_dim], activation=self.cfg.activation)

    def create_pos_emb(self):
        self.pos_emb = nn.Parameter(self.cfg.initializer_range*torch.randn(self.cfg.dim1, self.cfg.d_emodel))

    def get_encode_input(self, csi):
        csi = self.input_ffd_layer(csi.reshape(-1, self.cfg.dim1, self.cfg.dim2 * self.cfg.dim3))
        inputs = csi+self.pos_emb
        return inputs

    def get_encode_output(self, enc):
        enc = self.output_ffd_layer(enc.reshape(enc.shape[0], self.cfg.d_emodel*self.cfg.dim1))
        # enc = F.sigmoid(enc)
        return enc


class DecoderPlus(Decoder):
    def create_ffd(self):
        if not self.cfg.no_deq:
            self.input_ffd_layer = MLP(self.cfg.enc_dim, [self.cfg.d_dmodel*self.cfg.dim1], activation=self.cfg.activation)
        else:
            self.input_ffd_layer = MLP(self.cfg.enc_dim*self.cfg.n_q_bit, [self.cfg.d_dmodel*self.cfg.dim1], activation=self.cfg.activation)

        self.output_ffd_layer = MLP(self.cfg.d_dmodel, [self.cfg.dim2 * self.cfg.dim3], activation=self.cfg.activation)

    def create_pos_emb(self):
        self.pos_emb = nn.Parameter(self.cfg.initializer_range*torch.randn(self.cfg.dim1, self.cfg.d_dmodel))

    def get_decode_input(self, enc):
        enc = self.input_ffd_layer(enc)
        inputs = enc.reshape(enc.shape[0], self.cfg.dim1, self.cfg.d_dmodel)
        #inputs += self.pos_emb
        return inputs

    def get_decode_output(self, dec):
        dec = self.output_ffd_layer(dec)
        # dec = F.sigmoid(dec)
        return dec


# Note: Do not modify following class and keep it in your submission.
# feedback_bits is 512 by default.
class AutoEncoder(nn.Module):

    def __init__(self, feedback_bits):
        super(AutoEncoder, self).__init__()
        self.encoder = EncoderPlus(feedback_bits)
        self.decoder = DecoderPlus(feedback_bits)

    def forward(self, x):
        feature = self.encoder(x)
        out = self.decoder(feature)
        return out


def NMSE(x, x_hat):
    x_real = np.reshape(x[:, :, :, 0], (len(x), -1))
    x_imag = np.reshape(x[:, :, :, 1], (len(x), -1))
    x_hat_real = np.reshape(x_hat[:, :, :, 0], (len(x_hat), -1))
    x_hat_imag = np.reshape(x_hat[:, :, :, 1], (len(x_hat), -1))
    x_C = x_real - 0.5 + 1j * (x_imag - 0.5)
    x_hat_C = x_hat_real - 0.5 + 1j * (x_hat_imag - 0.5)
    power = np.sum(abs(x_C) ** 2, axis=1)
    mse = np.sum(abs(x_C - x_hat_C) ** 2, axis=1)
    nmse = np.mean(mse / power)
    return nmse


def Score(NMSE):
    score = 1 - NMSE
    return score


# dataLoader
class DatasetFolder(Dataset):

    def __init__(self, matData):
        self.matdata = matData

    def __getitem__(self, index):
        return self.matdata[index]

    def __len__(self):
        return self.matdata.shape[0]
