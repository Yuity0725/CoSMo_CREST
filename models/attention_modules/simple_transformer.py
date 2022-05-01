

import torch
import torch.nn as nn
from models.utils import reshape_text_features_to_concat
import math
import copy
import torch.nn.functional as F
from torch.autograd import Variable


def attention(q, k, v, d_k, mask=None, dropout=None):
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        mask = mask.unsqueeze(1)
        scores = scores.masked_fill(mask == 0, -1e9)
    scores = F.softmax(scores, dim=-1)

    if dropout is not None:
        scores = dropout(scores)

    output = torch.matmul(scores, v)
    return output


class Embedder(nn.Module):
    def __init__(self, feature_size, out_size):
        super().__init__()
        self.conv1 = nn.Conv2d(feature_size, out_size, kernel_size=7, bias=False)
        self.flatten = nn.Flatten()

    def forward(self, x):
        x = self.conv1(x)
        x = self.flatten(x)
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, heads, d_model, dropout=0.1):
        super().__init__()

        self.d_model = d_model
        self.d_k = d_model // heads
        self.h = heads

        self.q_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask=None):

        bs = q.size(0)

        # perform linear operation and split into h heads
        k = self.k_linear(k).view(-1, self.h, self.d_k)
        q = self.q_linear(q).view(-1, self.h, self.d_k)
        v = self.v_linear(v).view(-1, self.h, self.d_k)
        # transpose to get dimensions bs * h * sl * d_model

        k = k.transpose(1, 2)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)

        # calculate attention using function we will define next
        scores = attention(q, k, v, self.d_k, mask, self.dropout)

        # concatenate heads and put through final linear layer
        concat = scores.transpose(1, 2).contiguous()\
            .view(-1, self.d_model)

        output = self.out(concat)

        return output

class MultiHeadAttention2(nn.Module):
    def __init__(self, heads, d_model, dropout=0.1):
        super().__init__()

        self.d_model = d_model
        self.d_k = d_model // heads
        self.h = heads

        self.q_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, x, mask=None):

        bs = x.size(0)

        # perform linear operation and split into h heads
        k = self.k_linear(x).view(bs, -1, self.d_k, self.h)
        q = self.q_linear(x).view(bs, -1, self.d_k, self.h)
        v = self.v_linear(x).view(bs, -1, self.d_k, self.h)
        # transpose to get dimensions bs * h * sl * d_model
        k = k.transpose(1, 2)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)
        # calculate attention using function we will define next
        scores = attention(q, k, v, self.d_k, mask, self.dropout)
        # concatenate heads and put through final linear layer
        concat = scores.transpose(1, 2).contiguous()\
            .view(bs, -1, self.d_model)
        output = self.out(concat)

        return output





class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=2048, dropout=0.1):
        super().__init__()
        # We set d_ff as a default to 2048
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        x = self.dropout(F.relu(self.linear_1(x)))
        x = self.linear_2(x)
        return x


class Norm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()

        self.size = d_model
        # create two learnable parameters to calibrate normalisation
        self.alpha = nn.Parameter(torch.ones(self.size))
        self.bias = nn.Parameter(torch.zeros(self.size))
        self.eps = eps

    def forward(self, x):
        norm = self.alpha * (x - x.mean(dim=-1, keepdim=True)) \
            / (x.std(dim=-1, keepdim=True) + self.eps) + self.bias
        return norm


class EncoderLayer(nn.Module):
    def __init__(self, d_model, heads, dropout=0.1):
        super().__init__()
        self.norm_q = Norm(d_model)
        self.norm_k = Norm(d_model)
        self.norm_v = Norm(d_model)
        self.norm_1 = Norm(d_model)
        self.norm_2 = Norm(d_model)
        self.attn = MultiHeadAttention(heads, d_model)
        self.ff = FeedForward(d_model)
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)

    def forward(self, q, k, v):
        x = v
        q = self.norm_q(q)
        k = self.norm_k(k)
        v = self.norm_v(v)
        x = x + self.dropout_1(self.attn(q, k, v))
        x2 = self.norm_2(x)
        x = x + self.dropout_2(self.ff(x2))
        # x2 = self.norm_1(x)
        # x = x + self.dropout_1(self.attn(x2,x2,x2))
        # x2 = self.norm_2(x)
        # x = x + self.dropout_2(self.ff(x2))
        return x

class EncoderLayer2(nn.Module):
    """
    BERT ver
    """
    def __init__(self, d_model, heads, dropout=0.1):
        super().__init__()
        self.norm_1 = Norm(d_model)
        self.norm_2 = Norm(d_model)
        self.attn = MultiHeadAttention2(heads, d_model)
        self.ff = FeedForward(d_model)
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)

    def forward(self, x):
        input_x = self.norm_1(x)
        x = x + self.dropout_1(self.attn(input_x))
        x2 = self.norm_2(x)
        x = x + self.dropout_2(self.ff(x2))
        return x


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=2048, dropout=0.1):
        super().__init__()
        # We set d_ff as a default to 2048
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        x = self.dropout(F.relu(self.linear_1(x)))
        x = self.linear_2(x)
        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model, heads, dropout=0.1):
        super().__init__()
        self.norm_1 = Norm(d_model)
        self.norm_2 = Norm(d_model)
        self.norm_3 = Norm(d_model)

        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        self.dropout_3 = nn.Dropout(dropout)

        self.attn_1 = MultiHeadAttention2(heads, d_model)
        self.attn_2 = MultiHeadAttention2(heads, d_model)
        self.ff = FeedForward(d_model).cuda()

    def forward(self, x, e_outputs):
        x2 = self.norm_1(x)
        x = x + self.dropout_1(self.attn_1(x2, x2, e_outputs))
        x2 = self.norm_2(x)
        x = x + self.dropout_2(self.attn_2(x2, x2, e_outputs))
        x2 = self.norm_3(x)
        x = x + self.dropout_3(self.ff(x2))
        return x


def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class PositionalEncoder(nn.Module):
    def __init__(self, d_model, max_seq_len = 80):
        super().__init__()
        self.d_model = d_model
        
        # create constant 'pe' matrix with values dependant on 
        # pos and i
        pe = torch.zeros(max_seq_len, d_model)
        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = \
                math.sin(pos / (10000 ** ((2 * i)/d_model)))
                pe[pos, i + 1] = \
                math.cos(pos / (10000 ** ((2 * (i + 1))/d_model)))
                
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
 
    
    def forward(self, x):
        # make embeddings relatively larger
        x = x * math.sqrt(self.d_model)
        #add constant to embedding
        seq_len = x.size(1)
        x = x + Variable(self.pe[:,:seq_len], \
        requires_grad=False).cuda()
        return x

class Encoder(nn.Module):
    def __init__(self, d_model, N, heads):
        super().__init__()
        self.N = N
        # self.embed = Embedder(vocab_size, d_model)
        # self.pe = PositionalEncoder(d_model)
        self.layers = get_clones(EncoderLayer(d_model, heads), self.N)
        self.norm = Norm(d_model)

    def forward(self, q, k, x):
        # x = self.embed(src)
        # x = self.pe(x)
        for i in range(self.N):
            x = self.layers[i](q, k, x)
        return self.norm(x)

class Encoder2(nn.Module):
    """
    BERT ver
    """
    def __init__(self, d_model, N, heads):
        super().__init__()
        self.N = N
        # self.embed = Embedder(vocab_size, d_model)
        self.pe = PositionalEncoder(d_model)
        self.layers = get_clones(EncoderLayer2(d_model, heads), self.N)
        self.norm = Norm(d_model)

    def forward(self, x):
        # x = self.embed(src)
        x = self.pe(x)
        for i in range(self.N):
            x = self.layers[i](x)
        return self.norm(x)



class Decoder(nn.Module):
    def __init__(self, d_model, N, heads):
        super().__init__()
        self.N = N
        # self.embed = Embedder(vocab_size, d_model)
        self.pe = PositionalEncoder(d_model)
        self.layers = get_clones(DecoderLayer(d_model, heads), self.N)
        self.norm = Norm(d_model)

    def forward(self, trg, e_outputs):
        # x = self.embed(trg)
        x = self.pe(trg)
        # x = trg
        for i in range(self.N):
            x = self.layers[i](x, x)
        return self.norm(x)


class Transformer(nn.Module):
    def __init__(self, d_model, N, heads):
        super().__init__()
        self.encoder = Encoder(d_model, N, heads)
        self.out = nn.Linear(d_model, 49 * 49)

    def forward(self, q, k, v):
        e_outputs = self.encoder(q, k, v)
        output = self.out(e_outputs)
        return output

class Transformer2(nn.Module):
    """
    BERT ver
    """
    def __init__(self, d_model, N, heads):
        super().__init__()
        self.encoder = Encoder2(d_model, N, heads)
        # self.decoder = Decoder(d_model, N, heads)
        self.out = nn.Linear(d_model, 512)
    def forward(self, x):
        e_outputs = self.encoder(x)
        # d_outputs = self.decoder(e_outputs, e_outputs)
        output = self.out(e_outputs)
        return output[:, 0, :]
