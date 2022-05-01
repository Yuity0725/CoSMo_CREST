import torch
import torch.nn as nn

from models.utils import reshape_text_features_to_concat
from models.attention_modules.simple_transformer import Transformer, Transformer2, Embedder


class AttentionModule(nn.Module):
    def __init__(self, feature_size, text_feature_size, num_heads, *args, **kwargs):
        super().__init__()
        d_model = 512
        bert_size = 768
        self.n_heads = num_heads
        self.c_per_head = feature_size // num_heads
        assert feature_size == self.n_heads * self.c_per_head
        self.model = Transformer2(bert_size, 4, num_heads)

    def forward(self, x, bert, return_map=False, *args, **kwargs):
        img = x.view(-1, 1, bert.shape[2])
        vl_features = torch.cat([img, bert], dim=1)
        tra = self.model(vl_features)
        return tra, tra if return_map else tra

class SelfAttentionMap(nn.Module):
    def __init__(self, feature_size, num_heads, *args, **kwargs):
        super().__init__()

        self.n_heads = num_heads
        self.c_per_head = feature_size // num_heads
        assert feature_size == self.n_heads * self.c_per_head

        self.W_k = nn.Conv2d(feature_size, feature_size,
                             kernel_size=1, bias=False)
        self.W_q = nn.Conv2d(feature_size, feature_size,
                             kernel_size=1, bias=False)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, x, *args, **kwargs):
        b, c, h, w = x.size()

        keys, queries = self.W_k(x), self.W_q(x)
        # keys = keys.view(b * self.n_heads, self.c_per_head, h, w).view(b * self.n_heads, self.c_per_head, h * w)
        # queries = queries.view(b * self.n_heads, self.c_per_head, h, w).view(b * self.n_heads, self.c_per_head, h * w)

        keys = keys.view(b, self.n_heads * self.c_per_head, h * w)
        queries = queries.view(b, self.n_heads * self.c_per_head, h * w)

        att_map = torch.bmm(queries.transpose(1, 2), keys) / \
            (self.c_per_head ** 0.5)
        # (b * num_heads, h * w, h * w), torch.sum(att_map[batch_idx][?]) == 1
        att_map = self.softmax(att_map)
        # att_map = att_map.view(b, self.n_heads, h * w, h * w)

        return att_map


class GlobalCrossAttentionMap(nn.Module):
    def __init__(self, feature_size, text_feature_size, num_heads, normalizer=None, *args, **kwargs):
        super().__init__()

        self.n_heads = num_heads
        self.c_per_head = feature_size // num_heads
        assert feature_size == self.n_heads * self.c_per_head

        self.W_t = nn.Linear(text_feature_size, feature_size * 49)
        self.normalize = normalizer if normalizer else nn.Softmax(dim=1)

    def forward(self, x, t):
        b, c, h, w = x.size()

        # x_reshape = x.view(b * self.n_heads, self.c_per_head, h, w)
        # x_reshape = x_reshape.view(b * self.n_heads, self.c_per_head, h * w)

        x_reshape = x.view(b, self.n_heads * self.c_per_head, h * w)
        t_mapped = self.W_t(t).view(b, -1, 49)
        # t_mapped = t_mapped.view(b * self.n_heads, self.c_per_head, 1)
        att_map = torch.bmm(x_reshape.transpose(
            1, 2), t_mapped).squeeze(-1) / (self.c_per_head ** 0.5)
        att_map = self.normalize(att_map)  # (b * n_heads, h * w)
        # att_map = att_map.view(b * self.n_heads, 1, h * w)
        # att_map = att_map.view(b, self.n_heads, h * w)

        return att_map
