import torch
import torch.nn as nn
import torch.nn.functional as F

class LSTM(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, tagset_size, device):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.embed_dim = embedding_dim
        self.tagset_size = tagset_size

        # LSTM以word_embeddings作为输入, 输出维度为 hidden_dim 的隐藏状态值
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.device = device
        # 线性层将隐藏状态空间映射到标注空间

        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)
        self.hidden = self.init_hidden()

    def init_hidden(self):
        # 一开始并没有隐藏状态所以我们要先初始化一个
        # 关于维度为什么这么设计请参考Pytoch相关文档
        # 各个维度的含义是 (num_layers, minibatch_size, hidden_dim)
        return (torch.zeros(1, 1, self.hidden_dim).to(self.device),
                torch.zeros(1, 1, self.hidden_dim).to(self.device))

    def forward(self, sentence):
        lstm_out, self.hidden = self.lstm(
            sentence.view(1, -1, len(sentence)), self.hidden)
        # lstm_out, self.hidden = self.lstm(
        #     sentence.view(len(sentence), 1, -1), self.hidden)
        # lstm_out, self.hidden = self.lstm(
        #     sentence.view(1, -1, self.embed_dim), self.hidden)
        # tag_space = self.hidden2tag(lstm_out.view(len(sentence), -1))
        tag_space = self.hidden2tag(self.hidden[0])
        # tag_space = self.hidden2tag(self.hidden[0]).view(-1, self.tagset_size)
        # tag_scores = F.log_softmax(tag_space, dim=1)
        return tag_space