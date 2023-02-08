import torch.nn.functional as F
import torch
from torch import nn
import math

# 計算cosine similarity
def cosine(x,w):
    return F.linear(F.normalize(x,dim=-1), F.normalize(w,dim=-1))
# 也可以用其他metric如eucidien distance, 不過要配合不同的loss
def euc_dist(x,w):
    return F.pairwise_distance(x, w, 2)
    
# 計算loss
class ContrastiveLoss(nn.Module):
    def __init__(self,m=1):
        super().__init__()
        self.m=m
        self.activation=torch.sigmoid # 給cosine similarity用的，加強output contrast
        self.loss_fn=nn.BCELoss() # 給cosine similarity用的，加強output contrast
        self.z=torch.tensor(0.,dtype=torch.float32,requires_grad=False)
    def forward(self, y_pred,y_true):
        # 兩者同組時，算square
        # 兩者不同組時，算margin- distance值，若distance大於margin則不用再拉伸兩者distance
        print("shape = ",y_pred.shape)
        print("y_true = ",y_true.shape)
        loss=torch.mean(y_true * torch.square(y_pred)+ 
                        (1 - y_true)* torch.square(torch.maximum(self.m - y_pred, self.z)
                        ),dim=-1,keepdims=True)
        return loss.mean()
class FocalLoss(nn.Module):
    def __init__(self, gamma=2, eps=1e-10,ce=nn.CrossEntropyLoss()):
        super().__init__()
        self.gamma = gamma
        self.eps = torch.tensor(eps,dtype=torch.float32)
        self.ce = ce
    def forward(self,  y_pred,y_true):
        # 計算cross entropy
        L=self.ce(y_pred+self.eps, y_true)
        # 計算乘上gamma次方後的entropy反方機率(將對比放大)
        p = torch.exp(-L)
        loss = (1 - p) ** self.gamma * L
        return loss.mean()

class AddMarginLoss(nn.Module):
    def __init__(self, s=15.0, m=0.40,ways=10,loss_fn=FocalLoss()):
        super().__init__()
        self.s = s
        self.m = m
        self.ways=ways
        self.loss_fn=loss_fn
        
    def forward(self, cosine, label=None):
        # 扣掉對cosine的margin
        cos_phi = cosine - self.m
        # 將onehot沒選中的類別不套用margin，onehot選中的套用margin     
        one_hot=F.one_hot(label, num_classes=self.ways).to(torch.float32)
        metric = (one_hot * cos_phi) + ((1.0 - one_hot) * cosine)
        # 將輸出對比放大
        metric *= self.s
        return self.loss_fn(metric,label)

class ArcMarginLoss(AddMarginLoss):
    def __init__(self, s=32.0, m=0.40,ways=10, easy_margin=False,loss_fn=FocalLoss()):
        ## 使用AddMarginLoss的初始參數設定方式
        super().__init__(s,m,ways,loss_fn)
        
        ## 確定是否使用easy margin
        self.easy_margin = easy_margin
        ## 預先算好arc margin代表的cosine值、sine 值
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)

        # phi 在[0°,180°]以內的話，讓cos(phi+m)單調遞增
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m
        # 避免除以0發生，給一個輔助微小值
        self.eps = 1e-6
    def forward(self, cosine, label=None):
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2) + self.eps)
        # cos(phi)cos(m)-sin(phi)sin(m)變成cos(phi + m)
        # 這個margin加上去使得角度phi需要更小才能使指定類別在softmax(cos(phi))時最大
        cos_phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            # cosine如果不夠大就不用不套用phi margin
            cos_phi = torch.where(cosine > 0, cos_phi, cosine)
        else:
            # 更加嚴格，若cosine(phi)大於margin則套用phi margin規則
            #          若cosine(phi)小於margin則套用cosine margin規則
            cos_phi = torch.where(cosine > self.th, cos_phi, cosine - self.mm)
            
        # 將onehot沒選中的類別不套用margin，onehot選中的套用margin    
        one_hot=F.one_hot(label, num_classes=self.ways).to(torch.float32)
        metric = (one_hot * cos_phi) + ((1.0 - one_hot) * cosine)
        # 將輸出對比放大
        metric *= self.s
        return self.loss_fn(metric,label)

