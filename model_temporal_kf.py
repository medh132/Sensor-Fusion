import sys
import os.path as osp

import torch
import torch.nn as nn
import math
from torch.autograd import Variable

from tcn import TemporalConvNet


class TCNSeqNetwork(torch.nn.Module):
    def __init__(self, input_channel, output_channel, kernel_size, layer_channels, 
                 dropout=0.2, use_learnable_kf=False):
        """
        Temporal Convolution Network with PReLU activations
        Input: torch array [batch x frames x input_size]
        Output: torch array [batch x frames x out_size]

        :param input_channel: num. channels in input
        :param output_channel: num. channels in output
        :param kernel_size: size of convolution kernel (must be odd)
        :param layer_channels: array specifying num. of channels in each layer
        :param dropout: dropout probability
        :param use_learnable_kf: if True, adds learnable KF parameters
        """

        super(TCNSeqNetwork, self).__init__()
        self.kernel_size = kernel_size
        self.num_layers = len(layer_channels)
        self.use_learnable_kf = use_learnable_kf

        self.tcn = TemporalConvNet(input_channel, layer_channels, kernel_size, dropout)
        self.output_layer = torch.nn.Conv1d(layer_channels[-1], output_channel, 1)
        self.output_dropout = torch.nn.Dropout(dropout)
        self.net = torch.nn.Sequential(self.tcn, self.output_dropout, self.output_layer)
        
        # Learnable Kalman Filter parameters (in log space for positivity)
        if self.use_learnable_kf:
            self.log_q_pos = nn.Parameter(torch.tensor(-9.0))  # exp(-9) ≈ 1e-4
            self.log_q_vel = nn.Parameter(torch.tensor(-6.9))  # exp(-6.9) ≈ 1e-3
            self.log_r_vel = nn.Parameter(torch.tensor(-1.4))  # exp(-1.4) ≈ 0.25
        
        self.init_weights()

    def forward(self, x):
        out = x.transpose(1, 2)
        out = self.net(out)
        return out.transpose(1, 2)

    def init_weights(self):
        self.output_layer.weight.data.normal_(0, 0.01)
        self.output_layer.bias.data.normal_(0, 0.001)

    def get_receptive_field(self):
        return 1 + 2 * (self.kernel_size - 1) * (2 ** self.num_layers - 1)
    
    def get_kf_params(self):
        """
        Get current KF parameters (converted from log space)
        Returns None if not using learnable KF
        """
        if not self.use_learnable_kf:
            return None
        
        q_pos = torch.exp(self.log_q_pos)
        q_vel = torch.exp(self.log_q_vel)
        r_vel = torch.exp(self.log_r_vel)
        return q_pos, q_vel, r_vel


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512, dropout=0.1): #max_len=5000
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TransformerSeqNetwork(nn.Module):
    def __init__(self, input_channel, output_channel, d_model=128, nhead=4, 
                 num_layers=6, dim_feedforward=256, dropout=0.2, use_learnable_kf=False, causal=True):
        """
        Transformer Network for sequence processing
        Input: torch array [batch x frames x input_size]
        Output: torch array [batch x frames x out_size]
        
        :param input_channel: num. channels in input
        :param output_channel: num. channels in output
        :param d_model: dimension of transformer model (embedding size)
        :param nhead: number of attention heads
        :param num_layers: number of transformer layers
        :param dim_feedforward: dimension of feedforward network
        :param dropout: dropout probability
        :param use_learnable_kf: if True, adds learnable KF parameters
        """
        super(TransformerSeqNetwork, self).__init__()
        
        self.d_model = d_model
        self.num_layers = num_layers
        self.use_learnable_kf = use_learnable_kf
        self.causal = causal
        
        # Input projection
        self.input_projection = nn.Linear(input_channel, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='relu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output projection
        self.output_projection = nn.Linear(d_model, output_channel)
        self.output_dropout = nn.Dropout(dropout)
        
        # Learnable Kalman Filter parameters
        if self.use_learnable_kf:
            self.log_q_pos = nn.Parameter(torch.tensor(-9.0))
            self.log_q_vel = nn.Parameter(torch.tensor(-6.9))
            self.log_r_vel = nn.Parameter(torch.tensor(-1.4))
        
        self.init_weights()
    
    def init_weights(self):
        initrange = 0.1
        self.input_projection.weight.data.uniform_(-initrange, initrange)
        self.input_projection.bias.data.zero_()
        self.output_projection.weight.data.uniform_(-initrange, initrange)
        self.output_projection.bias.data.zero_()

    def generate_square_subsequent_mask(self, sz, device):
        """
        Generate causal mask for transformer
        Upper triangular matrix with True values (these positions are masked)
        
        :param sz: sequence length
        :param device: torch device
        :return: mask of shape [sz, sz]
        """
        mask = torch.triu(torch.ones(sz, sz, device=device), diagonal=1).bool()
        return mask
    
    def forward(self, x):
        # x: [batch, frames, input_channel]
        batch_size, seq_len, _ = x.shape
        x = self.input_projection(x)  # [batch, frames, d_model]
        x = self.pos_encoder(x)

        # Generate causal mask if needed
        if self.causal:
            mask = self.generate_square_subsequent_mask(seq_len, x.device)
            x = self.transformer_encoder(x, mask=mask)
        else:
            x = self.transformer_encoder(x)

        #x = self.transformer_encoder(x)  # [batch, frames, d_model]
        x = self.output_dropout(x)
        x = self.output_projection(x)  # [batch, frames, output_channel]
        return x
    
    def get_receptive_field(self):
        # Transformers have global receptive field
        return float('inf')
    
    def get_kf_params(self):
        """
        Get current KF parameters (converted from log space)
        Returns None if not using learnable KF
        """
        if not self.use_learnable_kf:
            return None
        
        q_pos = torch.exp(self.log_q_pos)
        q_vel = torch.exp(self.log_q_vel)
        r_vel = torch.exp(self.log_r_vel)
        return q_pos, q_vel, r_vel


class HybridTCNTransformer(nn.Module):
    def __init__(self, input_channel, output_channel, 
                 tcn_kernel_size=3, tcn_layer_channels=[64, 64], 
                 transformer_d_model=128, transformer_nhead=4,
                 dropout=0.2, combination='concat', use_learnable_kf=False):
        """
        Hybrid model combining TCN and Transformer
        
        :param input_channel: num. channels in input
        :param output_channel: num. channels in output
        :param tcn_kernel_size: TCN kernel size
        :param tcn_layer_channels: TCN layer channels
        :param transformer_d_model: Transformer model dimension
        :param transformer_nhead: Transformer attention heads
        :param dropout: dropout probability
        :param combination: how to combine outputs ('concat', 'add', 'weighted')
        :param use_learnable_kf: if True, adds learnable KF parameters
        """
        super(HybridTCNTransformer, self).__init__()
        
        self.combination = combination
        self.use_learnable_kf = use_learnable_kf
        
        # TCN branch        
        self.tcn = TemporalConvNet(input_channel, tcn_layer_channels, tcn_kernel_size, dropout)
        self.tcn_output = nn.Conv1d(tcn_layer_channels[-1], tcn_layer_channels[-1], 1)
        
        # Transformer branch
        self.transformer = TransformerSeqNetwork(
            input_channel, tcn_layer_channels[-1], 
            d_model=transformer_d_model,
            nhead=transformer_nhead,
            num_layers=6,
            dropout=dropout,
            use_learnable_kf=False  # Don't duplicate KF params
        )
        
        # Combination layer
        if combination == 'concat':
            in_dim = 2 * tcn_layer_channels[-1]
            self.combine = nn.Sequential(
                nn.Linear(in_dim, in_dim//2), 
                nn.ReLU(), 
                nn.Linear(in_dim//2, in_dim//4), 
                nn.ReLU(), 
                nn.Linear(in_dim//4, output_channel)
            )
        elif combination == 'weighted':
            self.alpha = nn.Parameter(torch.tensor(0.5))
        
        # Learnable Kalman Filter parameters
        if self.use_learnable_kf:
            self.log_q_pos = nn.Parameter(torch.tensor(-9.0))
            self.log_q_vel = nn.Parameter(torch.tensor(-6.9))
            self.log_r_vel = nn.Parameter(torch.tensor(-1.4))
        
        self.init_weights()
    
    def init_weights(self):
        self.tcn_output.weight.data.normal_(0, 0.01)
        if self.combination == 'concat':
            for layer in self.combine:
                if isinstance(layer, nn.Linear):
                    layer.weight.data.normal_(0, 0.01)
                    layer.bias.data.zero_()
    
    def forward(self, x):
        # x: [batch, frames, input_channel]
        
        # TCN branch
        tcn_out = x.transpose(1, 2)  # [batch, input_channel, frames]
        tcn_out = self.tcn(tcn_out)
        tcn_out = self.tcn_output(tcn_out)
        tcn_out = tcn_out.transpose(1, 2)  # [batch, frames, output_channel]
        
        # Transformer branch
        transformer_out = self.transformer(x)  # [batch, frames, output_channel]
        
        # Combine outputs
        if self.combination == 'concat':
            combined = torch.cat([tcn_out, transformer_out], dim=-1)
            output = self.combine(combined)
        elif self.combination == 'add':
            output = tcn_out + transformer_out
        elif self.combination == 'weighted':
            alpha = torch.sigmoid(self.alpha)
            output = alpha * tcn_out + (1 - alpha) * transformer_out
        else:
            raise ValueError(f"Unknown combination method: {self.combination}")
        
        return output
    
    def get_receptive_field(self):
        # Use TCN's receptive field for windowed operations
        kernel_size = 3  # Default, should match tcn_kernel_size
        num_layers = len([64, 64])  # Default, should match tcn_layer_channels
        return 1 + 2 * (kernel_size - 1) * (2 ** num_layers - 1)
    
    def get_kf_params(self):
        """
        Get current KF parameters (converted from log space)
        Returns None if not using learnable KF
        """
        if not self.use_learnable_kf:
            return None
        
        q_pos = torch.exp(self.log_q_pos)
        q_vel = torch.exp(self.log_q_vel)
        r_vel = torch.exp(self.log_r_vel)
        return q_pos, q_vel, r_vel
