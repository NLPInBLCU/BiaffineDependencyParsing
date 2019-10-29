import torch
import torch.nn as nn
import torch.nn.functional as F


class PairwiseBilinear(nn.Module):
    '''
    使用版本
    A bilinear module that deals with broadcasting for efficient memory usage.
    Input: tensors of sizes (N x L1 x D1) and (N x L2 x D2)
    Output: tensor of size (N x L1 x L2 x O)'''

    def __init__(self, input1_size, input2_size, output_size, bias=True):
        super().__init__()

        self.input1_size = input1_size
        self.input2_size = input2_size
        self.output_size = output_size
        # W size: [(head_fea_size+1),(dep_fea_size+1),output_size]
        # 无标签弧分类时 output_size=1
        # 标签分类时 output_size=len(labels)
        self.weight = nn.Parameter(torch.Tensor(input1_size, input2_size, output_size))
        self.bias = nn.Parameter(torch.Tensor(output_size)) if bias else 0

    def forward(self, input1, input2):
        input1_size = list(input1.size())
        input2_size = list(input2.size())
        output_size = [input1_size[0], input1_size[1], input2_size[1], self.output_size]

        # ((N x L1) x D1) * (D1 x (D2 x O)) -> (N x L1) x (D2 x O)
        # [(batch_size*seq_len),(head_feat_size+1)] * [(head_feat_size+1),((dep_feat_size+1))*output_size]
        # -> [(batch_size*seq_len),((dep_feat_size+1))*output_size]
        intermediate = torch.mm(input1.view(-1, input1_size[-1]),
                                self.weight.view(-1, self.input2_size * self.output_size))
        # (N x L2 x D2) -> (N x D2 x L2)
        # input2 size: [batch_size, (dep_feat_size+1), seq_len]
        input2 = input2.transpose(1, 2)
        # (N x (L1 x O) x D2) * (N x D2 x L2) -> (N x (L1 x O) x L2)
        # intermediate size:
        # [(batch_size*seq_len),((dep_feat_size+1))*output_size]
        #       ->[batch_size, (seq_len*output_size), (dep_feat_size+1)]

        # [batch_size, (seq_len*output_size), (dep_feat_size+1)] * [batch_size, (dep_feat_size+1), seq_len]
        # -> [batch_size, (seq_len*output_size), seq_len]
        output = intermediate.view(input1_size[0], input1_size[1] * self.output_size, input2_size[2]).bmm(input2)
        # (N x (L1 x O) x L2) -> (N x L1 x L2 x O)
        # output size: [batch_size, seq_len, seq_len, output_size]
        output = output.view(input1_size[0], input1_size[1], self.output_size, input2_size[1]).transpose(2, 3)

        return output


class BiaffineScorer(nn.Module):
    def __init__(self, input1_size, input2_size, output_size):
        super().__init__()
        # 为什么+1？？
        # 双仿变换的矩阵形式：
        # S=(H_head⊕1)·W·H_dep
        # 即：(d*d) = (d*(k+1)) * ((k+1)*k) * (k*d)
        self.W_bilin = nn.Bilinear(input1_size + 1, input2_size + 1, output_size)

        self.W_bilin.weight.data.zero_()
        self.W_bilin.bias.data.zero_()

    def forward(self, input1, input2):
        # input1 size：[batch_size, seq_len, feature_size]
        # input1.new_ones(*input1.size()[:-1], 1)'s size: [batch_size, seq_len, 1]
        input1 = torch.cat([input1, input1.new_ones(*input1.size()[:-1], 1)], len(input1.size()) - 1)
        # 拼接后的size:[batch_size, seq_len, (feature_size+1)]
        input2 = torch.cat([input2, input2.new_ones(*input2.size()[:-1], 1)], len(input2.size()) - 1)
        return self.W_bilin(input1, input2)


class PairwiseBiaffineScorer(nn.Module):
    def __init__(self, input1_size, input2_size, output_size):
        """
        使用版本
        :param input1_size:
        :param input2_size:
        :param output_size:双仿的分类空间
        """
        super().__init__()
        # 为什么+1:
        # 双仿变换的矩阵形式：
        # [(batch_size*seq_len),(head_feat_size+1)] * [(head_feat_size+1),((dep_feat_size+1))*output_size]
        #       mm-> [(batch_size*seq_len),((dep_feat_size+1))*output_size]
        # [(batch_size*seq_len),((dep_feat_size+1))*output_size]
        #       view-> [batch_size, (seq_len*output_size), (dep_feat_size+1)]
        # [batch_size, (seq_len*output_size), (dep_feat_size+1)] * [batch_size, (dep_feat_size+1), seq_len]
        #       bmm-> [batch_size, (seq_len*output_size), seq_len]
        # [batch_size, (seq_len*output_size), seq_len]
        #       view-> [batch_size, seq_len, seq_len, output_size]
        self.W_bilin = PairwiseBilinear(input1_size + 1, input2_size + 1, output_size)

        self.W_bilin.weight.data.zero_()
        self.W_bilin.bias.data.zero_()

    def forward(self, input1, input2):
        # input1 size：[batch_size, seq_len, feature_size]
        # input1.new_ones(*input1.size()[:-1], 1)'s size: [batch_size, seq_len, 1]
        input1 = torch.cat([input1, input1.new_ones(*input1.size()[:-1], 1)], len(input1.size()) - 1)
        # 拼接后的size:[batch_size, seq_len, (feature_size+1)]
        input2 = torch.cat([input2, input2.new_ones(*input2.size()[:-1], 1)], len(input2.size()) - 1)
        return self.W_bilin(input1, input2)


class DirectBiaffineScorer(nn.Module):
    def __init__(self, input1_size, input2_size, output_size, pairwise=True):
        super().__init__()
        if pairwise:
            self.scorer = PairwiseBiaffineScorer(input1_size, input2_size, output_size)
        else:
            self.scorer = BiaffineScorer(input1_size, input2_size, output_size)

    def forward(self, input1, input2):
        return self.scorer(input1, input2)


class DeepBiaffineScorer(nn.Module):
    def __init__(self, input1_size, input2_size, hidden_size, output_size, hidden_func=F.relu, dropout=0,
                 pairwise=True):
        """
        使用版本
        :param input1_size:
        :param input2_size:
        :param hidden_size:
        :param output_size: 双仿的分类空间
        :param hidden_func:
        :param dropout:
        :param pairwise:
        """
        super().__init__()
        # 先对输入做两个线性变换得到两个H_dep、H_head
        self.W1 = nn.Linear(input1_size, hidden_size)
        self.W2 = nn.Linear(input2_size, hidden_size)
        # 默认经过relu激活函数：
        self.hidden_func = hidden_func
        if pairwise:
            self.scorer = PairwiseBiaffineScorer(hidden_size, hidden_size, output_size)
        else:
            self.scorer = BiaffineScorer(hidden_size, hidden_size, output_size)
        # 进入双仿前dropout:
        self.dropout = nn.Dropout(dropout)

    def forward(self, input1, input2):
        return self.scorer(self.dropout(self.hidden_func(self.W1(input1))),
                           self.dropout(self.hidden_func(self.W2(input2))))


if __name__ == "__main__":
    x1 = torch.randn(2, 3, 4)
    x2 = torch.randn(2, 3, 5)
    scorer = DeepBiaffineScorer(4, 5, 6, 7)
    print(scorer(x1, x2))
    res = scorer(x1, x2)
    print(res.size())
