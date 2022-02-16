import os, sys, logging
sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))
import numpy as np
import logging
from torch import nn
import json

import torch
import torch.nn.functional as F
from torch.nn.init import xavier_normal_
from copy import deepcopy
import math

gl = globals()
NUM_FEEDBACK_BITS = 128


logger = logging.getLogger(__name__)

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
    #num = torch.zeros(Bit_[:, :, 1].shape, device=Bit.device)
    num = torch.zeros(Bit_[:, :, 0].shape, device=Bit.device)
    for i in range(B):
        num = num + Bit_[:, :, i] * 2 ** (B - 1 - i)
    return num


class Quantization(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, B):
        ctx.constant = B
        #step = 2 ** B
        #out = torch.round(x * step - 0.5)
        out = torch.round(x)
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
        #step = 2 ** B
        out = Bit2Num(x, B)
        #out = (out + 0.5) / step
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

class MixQuantizationLayer(nn.Module):
    def __init__(self, cfg, dim):
        super().__init__()
        self.cfg = cfg
        self.dim1 = round(dim*self.cfg.mixq_factor)
        assert self.dim1>0
        logger.info('mixquan, dim %s, mix dim is:%s', dim, self.dim1)
        self.dim2 = dim-self.dim1
        self.q1 = QuantizationLayer(self.cfg.n_q_bit-1)
        self.q2 = QuantizationLayer(self.cfg.n_q_bit)

    def forward(self, x):
        x1 = x[..., :self.dim1]
        x2 = x[..., self.dim1:]
        x1 = self.q1(x1)
        x2 = self.q2(x2)
        x = torch.cat([x1, x2], -1)
        return x


class DequantizationLayer(nn.Module):
    def __init__(self, B):
        super(DequantizationLayer, self).__init__()
        self.B = B

    def forward(self, x):
        out = Dequantization.apply(x, self.B)
        return out

class MixDequantizationLayer(nn.Module):
    def __init__(self, cfg, dim):
        super().__init__()
        self.cfg = cfg
        self.dim1 = round(dim*self.cfg.mixq_factor)
        assert self.dim1>0
        logger.info('mixdeq dim:%s mix dim is:%s',dim, self.dim1)
        self.dim2 = dim-self.dim1
        self.dq1 = DequantizationLayer(self.cfg.n_q_bit-1)
        self.dq2 = DequantizationLayer(self.cfg.n_q_bit)

    def forward(self, x):
        x1 = x[..., :self.dim1*(self.cfg.n_q_bit-1)]
        x2 = x[..., self.dim1*(self.cfg.n_q_bit-1):]
        x1 = self.dq1(x1)
        x2 = self.dq2(x2)
        x = torch.cat([x1, x2], -1)
        return x


def cross_entropy(logits, labels, mask):
    logits = torch.transpose(logits, 1, len(logits.size())-1)
    loss = torch.sum(F.cross_entropy(logits, labels, reduction='none')*mask)/torch.sum(mask)
    return loss


def binary_cross_entropy(logits, labels, mask):

    loss = torch.sum(F.binary_cross_entropy_with_logits(logits, labels.to(logits.dtype), reduction='none')*mask)/torch.sum(mask)
    return loss


def gelu_new(x):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT). Also see
    the Gaussian Error Linear Units paper: https://arxiv.org/abs/1606.08415
    """
    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))


def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu

    raise RuntimeError("activation should be relu/gelu, not {}".format(activation))

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
        # self.output_ffd_layer = MLP(self.cfg.d_emodel*self.cfg.dim1, [self.cfg.enc_dim], activation=self.cfg.activation)


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
        # csi = (csi-self.mean_v)/self.std_v
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

# class EncoderPlus(Encoder):
#     def create_ffd(self):
#         self.input_ffd_layer = MLP(self.cfg.dim2 * self.cfg.dim3, [self.cfg.d_emodel], activation=self.cfg.activation)
#         self.output_ffd_layer = MLP(self.cfg.d_emodel*self.cfg.dim1, [self.cfg.enc_dim], activation=self.cfg.activation)

#     def create_pos_emb(self):
#         self.pos_emb = nn.Parameter(self.cfg.initializer_range*torch.randn(self.cfg.dim1, self.cfg.d_emodel))

#     def get_encode_input(self, csi):
#         csi = self.input_ffd_layer(csi.reshape(-1, self.cfg.dim1, self.cfg.dim2 * self.cfg.dim3))
#         inputs = csi+self.pos_emb
#         return inputs

#     def get_encode_output(self, enc):
#         enc = self.output_ffd_layer(enc.reshape(enc.shape[0], self.cfg.d_emodel*self.cfg.dim1))
#         enc = F.sigmoid(enc)
#         return enc


# class DecoderPlus(Decoder):
#     def create_ffd(self):
#         if not self.cfg.no_deq:
#             self.input_ffd_layer = MLP(self.cfg.enc_dim, [self.cfg.d_dmodel*self.cfg.dim1], activation=self.cfg.activation)
#         else:
#             self.input_ffd_layer = MLP(self.cfg.enc_dim*self.cfg.n_q_bit, [self.cfg.d_dmodel*self.cfg.dim1], activation=self.cfg.activation)

#         self.output_ffd_layer = MLP(self.cfg.d_dmodel, [self.cfg.dim2 * self.cfg.dim3], activation=self.cfg.activation)

#     def create_pos_emb(self):
#         self.pos_emb = nn.Parameter(self.cfg.initializer_range*torch.randn(self.cfg.dim1, self.cfg.d_dmodel))

#     def get_decode_input(self, enc):
#         enc = self.input_ffd_layer(enc)
#         inputs = enc.reshape(enc.shape[0], self.cfg.dim1, self.cfg.d_dmodel)
#         #inputs += self.pos_emb
#         return inputs

#     def get_decode_output(self, dec):
#         dec = self.output_ffd_layer(dec)
#         # dec = F.sigmoid(dec)
#         return dec


class CSI(Module):
    def create_layers(self):
        self.encoder = gl[self.cfg.encoder](self.cfg)
        if self.cfg.decoder.endswith('P') or self.cfg.decoder.endswith('Plus'):
            self.decoder = gl[self.cfg.decoder](self.cfg, encoder=self.encoder)
        else:
            self.decoder = gl[self.cfg.decoder](self.cfg)
        if self.cfg.distill_model is not None:
            self.distill_eproj = nn.Linear(self.cfg.d_emodel, self.cfg.distill_d_emodel)
            self.distill_dproj = nn.Linear(self.cfg.d_dmodel, self.cfg.distill_d_dmodel)


    def forward(self, csi, **kwargs):
        #csi = csi.permute(0, 2, 3, 1)
        enc = self.encoder(csi, **kwargs)
        enc_preq = enc
        if not self.cfg.is_test and not self.cfg.encoder.endswith('P'):
            enc, enc_preq = enc
        dec = self.decoder(enc, **kwargs)
        #dec = dec.permute(0, 3, 1, 2)
        outputs = {'dec': dec, 'enc_preq':enc_preq}
        return outputs

    def calc_loss(self, inputs, outputs):
        # nmse_loss=NMSE(inputs['csi'],outputs['dec'])
        csi = inputs['csi']-0.5
        dec = outputs['dec']-0.5
        enc_preq = outputs['enc_preq']
        power_csi = csi[:, :, :, 0]**2 + csi[:, :, :, 1]**2
        diff = csi - dec
        power_diff = diff[:, :, :, 0]**2 + diff[:, :, :, 1]**2
        nmse_loss = torch.mean(torch.sum(power_diff,[1,2])/torch.sum(power_csi, [1,2]))
        losses = {'nmse_loss':nmse_loss}
        if self.cfg.use_mse:
            mse_loss = F.mse_loss(csi, dec)
            losses['mse_loss'] = mse_loss
            loss = mse_loss
        else:
            loss = nmse_loss
        losses['loss'] = loss

        if self.cfg.use_round_loss:
            round_enc = torch.round(enc_preq)
            round_enc = round_enc.detach()
            round_loss = F.mse_loss(enc_preq, round_enc)
            losses['loss'] = losses['loss'] + self.cfg.round_weight*round_loss
            losses['round_loss'] = round_loss
        if self.cfg.distill_model is not None and self.training:
            eh_loss = F.mse_loss(self.distill_eproj(self.encoder.encoder.enc_hidden), inputs['enc_hidden'][:, ::self.cfg.distill_factor])
            dh_loss = F.mse_loss(self.distill_dproj(self.decoder.decoder.enc_hidden), inputs['dec_hidden'][:, ::self.cfg.distill_factor])
            ea_loss = F.mse_loss(self.encoder.encoder.enc_att, inputs['enc_att'][:, ::self.cfg.distill_factor])
            da_loss = F.mse_loss(self.decoder.decoder.enc_att, inputs['dec_att'][:, ::self.cfg.distill_factor])
            losses['rep_loss'] = ea_loss + da_loss + eh_loss + dh_loss
            losses['enc_loss'] = F.mse_loss(self.encoder.enc_enc, inputs['enc_enc'])
            losses['loss'] = losses['loss'] + losses['rep_loss'] + losses['enc_loss']
        return losses

class GAN(CSI):
    def create_layers(self):
        pass

    def forward(self, csi, **kwargs):
        pass

    def calc_loss(self, inputs, outputs):
        pass

class CFG(object):
    pass
class AutoEncoder(nn.Module):
    def __init__(self, feedback_bits):
        super(AutoEncoder, self).__init__()
        cfg_fpath = os.path.join(os.path.dirname(__file__), 'cfg.json')
        with open(cfg_fpath) as f:
            cfg = json.load(f)
        self.cfg = CFG()
        for k, v in cfg.items():
            setattr(self.cfg, k, v)
        self.cfg.ema = 0.0
        self.cfg.is_test = True
        self.encoder = gl[self.cfg.encoder](self.cfg)
        self.decoder = gl[self.cfg.decoder](self.cfg, encoder=self.encoder)
        if self.cfg.do_dyq:
            self.encoder = torch.quantization.quantize_dynamic(self.encoder, {torch.nn.Linear}, dtype=torch.qint8)
            self.decoder = torch.quantization.quantize_dynamic(self.decoder, {torch.nn.Linear}, dtype=torch.qint8)

    def forward(self, x):
        feature = self.encoder(x)
        out = self.decoder(feature)
        return out


class DatasetFolder(torch.utils.data.Dataset):
    def __init__(self, matData):
        self.matdata = matData
    def __getitem__(self, index):
        return self.matdata[index]
    def __len__(self):
        return self.matdata.shape[0]

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

def NMSE_cuda1(x, x_hat):
    x_real = x[:, :, :, 0].view(len(x),-1) - 0.5
    x_imag = x[:, :, :, 1].view(len(x),-1) - 0.5
    x_hat_real = x_hat[:, :, :, 0].view(len(x_hat), -1) - 0.5
    x_hat_imag = x_hat[:, :, :, 1].view(len(x_hat), -1) - 0.5
    power = torch.sum(x_real**2 + x_imag**2, axis=1)
    mse = torch.sum((x_real-x_hat_real)**2 + (x_imag-x_hat_imag)**2, axis=1)
    nmse = mse/power
    return nmse

if __name__ == "__main__":
    gl = globals()


