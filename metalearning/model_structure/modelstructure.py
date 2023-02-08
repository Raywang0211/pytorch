from torch import nn
import torch
import torch.nn.functional as F


def cosine(x,w):
    return F.linear(F.normalize(x,dim=-1), F.normalize(w,dim=-1))
# 也可以用其他metric如eucidien distance, 不過要配合不同的loss
def euc_dist(x,w):
    return F.pairwise_distance(x, w, 2)


# convolution cell
class conv(nn.Module):
    def __init__(self,ch_in,ch_out):
        super().__init__()
        self.cell=nn.Sequential(
            nn.Conv2d(ch_in, ch_out, 3, 1, 1),
            nn.InstanceNorm2d(ch_out),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.2)
        )
    def forward(self,x):
        return self.cell(x)
    
# dense cell: 
class dense(nn.Module):
    def __init__(self,ch_in=512,ch_out=512,squeeze_input=False):
        super().__init__()
        self.cell=nn.Sequential(
            nn.Linear(ch_in, ch_out),
            nn.BatchNorm1d(ch_out),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        self.squeeze_input=squeeze_input
    def forward(self,x):
        if self.squeeze_input:
            return self.cell(x.squeeze()) #功能類似flatten 並且把不用的rank 刪掉
        return self.cell(x)

class MetricLayer(nn.Module):
    def __init__(self, n_in_features,n_out_features=10,metric=cosine):
        super().__init__()
        # 做一些參數儲存prototype位置，並給予random初始化
        self.weight = nn.Parameter(torch.Tensor(n_out_features, n_in_features))
        nn.init.xavier_uniform_(self.weight,gain=1.0)
        # 指定要比較時用到的metric
        self.metric=metric
    def forward(self,x):
        # 使用時將輸入query的latent與前面prototype計算metric
        return self.metric(x,self.weight)


class FeatureExtractor(nn.Module):
    def __init__(self,
                 in_features=3,
                 latent_features=3,
                 hidden_featrues=(64,128,256)):
        super().__init__()
        self.blocks = nn.Sequential(
            conv(in_features,hidden_featrues[0]),
            *[conv(ch1, ch2) for ch1,ch2 in zip(hidden_featrues[:-1],hidden_featrues[1:])],
            nn.AdaptiveAvgPool2d((1,1)), #gloabe average pooling<用來取代flatten>
            dense(hidden_featrues[-1],latent_features,squeeze_input=True)
        )
    def forward(self, x):
        return self.blocks(x)


class ClasModel(nn.Module):
    def __init__(self,ways,shots,backbone,head,metric=cosine):
        super().__init__()
        self.ways=ways
        self.shots=shots

        self.backbone=backbone
        self.head=head
        self.metric=metric

class SiameseNet(ClasModel):
    def __init__(self,backbone,metric=cosine):
        super().__init__(2,None,backbone,None,metric)
    def forward(self,data,label=None):
        # 進Embedding
        latent=[*map(self.backbone,data.transpose(0,1))]
        # latent算metric 這邊用cosine
        logits=torch.stack([*map(self.metric,latent[0],latent[1])],dim=0)
        return logits


class Base(ClasModel):
    def __init__(self,ways,backbone,head):
        assert(backbone is not None)
        # 透過ClasModel這個parent來initail 一些class attribute
        # 包含way數、backbone module、head module
        super().__init__(ways,None,backbone,head)
    def forward(self,data,label=None):
        # Transfer Learing: backbone + output head
        hidden=self.backbone(data) # 將data放進backbone生成latent vector
        logits=self.head(hidden)   # latent vector 丟進output later做最終classifcation
        return logits