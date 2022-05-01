from torch import nn

from models.utils import calculate_mean_std, EqualLinear
from trainers.abc import AbstractGlobalStyleTransformer


class GlobalStyleTransformer2(AbstractGlobalStyleTransformer):
    def __init__(self, feature_size, text_feature_size, *args, **kwargs):
        super().__init__()
        self.feature_size = feature_size
        self.global_transform = EqualLinear(text_feature_size, feature_size * 2)
        self.gate = EqualLinear(text_feature_size, feature_size * 2)
        self.sigmoid = nn.Sigmoid()
        self.flatten = nn.Flatten()
        self.init_style_weights(feature_size)
        self.linear = nn.Linear(2048, 768)
        self.linear2 = nn.Linear(768, 2048)
        
    def forward(self, normed_x, t, bert, *args, **kwargs):
        x_mu, x_std = calculate_mean_std(kwargs['x'])
        bert_out = normed_x
        bert_out = bert_out.view(-1, 1, 768)
        bert_mu, bert_std = self.linear(self.flatten(x_mu)).unsqueeze(-1).transpose(1, 2), self.linear(self.flatten(x_std)).unsqueeze(-1).transpose(1, 2)
        # gate = self.sigmoid(self.gate(t)).unsqueeze(-1).unsqueeze(-1)
        # std_gate, mu_gate = gate.chunk(2, 1)

        # global_style = self.global_transform(t).unsqueeze(2).unsqueeze(3)
        # gamma, beta = global_style.chunk(2, 1)

        # gamma = std_gate * x_std + gamma
        # beta = mu_gate * x_mu + beta
        bert_gamma = bert * bert_std
        bert_beta = bert * bert_mu
        tmpout = bert_gamma * bert_out + bert_beta * bert_out
        # out = gamma * normed_x + beta
        tmpout = self.linear2(tmpout).transpose(1, 2)
        tmpout = nn.Linear(tmpout.shape[2], 49).to("cuda")(tmpout).view(-1, 2048, 7, 7)
        # out = gamma * normed_x + beta * normed_x + gamma + beta
        return tmpout

    def init_style_weights(self, feature_size):
        self.global_transform.linear.bias.data[:feature_size] = 1
        self.global_transform.linear.bias.data[feature_size:] = 0

    @classmethod
    def code(cls) -> str:
        return 'global2'
