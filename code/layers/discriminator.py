import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, n_h):   ### n_h是embedding设定的维度
        super(Discriminator, self).__init__()
        self.f_k = nn.Bilinear(n_h, n_h, 1)   ##双线性变换层，最后输出的维度为1

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Bilinear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, c, h_pl, h_mi, s_bias1=None, s_bias2=None): ## 在dgi.py中，c为全局表示，h_pl为正样本局部节点表示，h_mi为负样本节点表示
        # print("c 为：{}".format(c.size()))
        c_x = torch.unsqueeze(c, 1) ##维度由[1, 128]变成[1, 1, 128]
        c_x = c_x.expand_as(h_pl)  ##将张量c_x扩张到h_p1的尺寸


        sc_1 = torch.squeeze(self.f_k(h_pl, c_x), 2)  ##根据self.k的定义，c_x的大小为n_h
        sc_2 = torch.squeeze(self.f_k(h_mi, c_x), 2)  ##根据self.k的定义，c_x的大小为n_h

        if s_bias1 is not None:
            sc_1 += s_bias1
        if s_bias2 is not None:
            sc_2 += s_bias2

        # logits = torch.cat((sc_1, sc_2), 1)   ##logits的维度为[1, 2*节点数量]
        # return logits
        return sc_1, sc_2

