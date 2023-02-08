import os
from glob import glob
import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
from tqdm.notebook import tqdm
from sklearn.metrics import confusion_matrix,classification_report

# Backend: pytorch
import torch
import torchvision.datasets as tvds 
import torchvision.transforms as tvt #用來進行augmentation 用的
import torch.utils.data as tud
from torch import nn
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader
from torchsummary import summary
import torch.optim as optim

# 設一個 show data和label的function
def show_data(imgs, titles,cmap=None):
    # imshow, 縮放模式為nearest。
    plt.figure(figsize=(18, 18))
    for id,(img,title) in enumerate(zip(imgs,titles)):
        plt.subplot(1, len(titles), id+1)
        fig = plt.imshow(img,interpolation='nearest',cmap=cmap)
        plt.axis("off")
        plt.title(title)
def train_one_epoch(epoch,model,loss_function,optimizer,dataloader,adapt=False):
    running_loss =0.0
    total_hit,total_num = 0.0,0.0
    session = enumerate(dataloader)
    
    print("epoch= ",epoch)
    print("training||",end='')
    for i,(data,label) in session:
        data = data.cuda()
        label =label.cuda()
        if not adapt:
            class_logits = model(data)
        else:
            with torch.no_grad():
                latent = model.backbone(data)
            class_logits = model.head(latent.detach())
        losses = loss_function(class_logits,label)
        losses.backward()
        optimizer.step()
        optimizer.zero_grad()

        running_loss +=losses.item()


        total_hit += torch.sum(torch.argmax(class_logits,dim=1)==label).item()
        total_num += data.shape[0]

            
            
 
    return running_loss/(i+1), total_hit/total_num

def run_test(model,dataloder):
    result = dict(y_ture=[],y_pre=[])
    total_test_hit,total_test_num = 0.0,0.0
    with torch.no_grad():
        for i,(data,label) in enumerate(dataloder):
            test_data = data.cuda()
            test_label = label.cuda()
            class_logits = model(test_data)
            pred = torch.argmax(class_logits, dim=1).cpu().detach().numpy()
            result["y_ture"].extend(label.cpu().detach().numpy())
            result["y_pre"].extend(pred)
            total_test_hit += torch.sum(torch.argmax(class_logits, dim=1) == test_label).item()
            total_test_num += test_data.shape[0]

    print("test acc:",total_test_hit / total_test_num)
    return result,total_test_hit / total_test_num




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

class Baseline(ClasModel):
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






class FewShotSampler(tud.Sampler):
    def __init__(self,ds_object,classes,shots,repeats=16,shuffle=True):
        # 儲存參數進object
        self.ds_object=ds_object
        self.classes=classes
        self.shots=shots
        self.repeats=repeats # 在dataset總數少的時候，為了可以組batch，會把多個epoch的資料疊在一起
        self.class_samples=self.get_samples()
        self.shuffle=shuffle
        
    def get_samples(self):
        # 將dataset每個內容拿出來，若class與指定clas相同則存起來
        new_target_list=np.random.permutation([*enumerate(self.ds_object.targets)])

        indices=[]
        for c in self.classes:
            count=self.shots
            for ii,yy in new_target_list:
                # 若已抽到 shot個則停止
                if count==0:
                    break
                # 還未抽到則檢查該label是否為指定class
                if yy==self.ds_object.class_to_idx[c]:
                    # 若是，則加入列表
                    indices.append(ii)
                    count-=1
        return np.repeat(indices,self.repeats)
    
    def __len__(self):
        return len(self.class_samples)
    def __iter__(self):
        if self.shuffle:
            return iter(np.random.permutation(self.class_samples))
        return iter(self.class_samples)






# 看source data: 
titles = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
example_files=[glob(f"/code/mnistm_half/mnist_test/{i}/*.png")[0] for i in titles]
show_data([*map(plt.imread,example_files)],titles)
example_files=[glob(f"/code/mnistm_half/mnist_m_train/{i}/*.png")[0] for i in titles]
show_data([*map(plt.imread,example_files)],titles)

# 生成一個 augmnetation function
train_transform = tvt.Compose([
    tvt.RandomHorizontalFlip(),
    # 旋轉15度內 (Augmentation)，旋轉後空的地方補0
    tvt.RandomRotation(15, fill=(0,0,0)),
    tvt.ToTensor()
])
test_transform = tvt.Compose([
    tvt.ToTensor()
])
source_datasets = tvds.ImageFolder("/code/mnistm_half/mnist_m_train",transform=train_transform)
target_datasets = tvds.ImageFolder("/code/FPC/train",transform=train_transform)
test_dataset = tvds.ImageFolder("/code/FPC/test",transform=test_transform)

WAYS=10
SHOTS=10
BS_TRAIN = 5
BS_TEST = 1
LATEN_SPACE = 30
sampler = FewShotSampler(target_datasets,target_datasets.classes,SHOTS,repeats=100)
source_dataloader = DataLoader(source_datasets,
                                batch_size=BS_TRAIN,
                                shuffle=True,
                                num_workers=0,
                                pin_memory=True)
target_dataloader = DataLoader(target_datasets,
                                batch_size=WAYS*SHOTS,
                                num_workers=0,
                                pin_memory=True,
                                sampler=sampler)
target_test_loader = DataLoader(test_dataset,
                                batch_size=BS_TEST,
                                shuffle=False,
                                num_workers=0,
                                pin_memory=True)

backbone = FeatureExtractor(latent_features=LATEN_SPACE)

model_type = "baseline++"

if model_type=="baseline":
    head = nn.Linear(LATEN_SPACE,WAYS)
elif model_type=="baseline++":
    head=MetricLayer(LATEN_SPACE,WAYS,cosine)
else:
    print("No model")

model = Baseline(WAYS,backbone=backbone,head=head).cuda()
model.load_state_dict(torch.load("baseline++_focal_FPC.pth"))

loss_type = "arcmargin"
if loss_type == "cel":
    loss_function = nn.CrossEntropyLoss()
elif loss_type == "focal":
    loss_function = FocalLoss()
elif loss_type == "arcmargin":
    loss_function = ArcMarginLoss(ways=WAYS,loss_fn=FocalLoss())
elif loss_type == "addmargin":
    loss_function = AddMarginLoss(ways=WAYS,loss_fn=FocalLoss())
else:
    print("No loss function")


LR = 0.001
optimizer = optim.Adam(model.parameters(),lr=LR)
EPOCH = 200
patience = 20
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode="min",factor=0.1,patience=5)

try:
    temp_train_loss = 999999
    loss_patience = 0
    for epoch in range(EPOCH):
        train_loss,train_acc = train_one_epoch(epoch=epoch,
                                                model=model,
                                                loss_function=loss_function,
                                                optimizer=optimizer,
                                                dataloader=source_dataloader,
                                                adapt=False)

        print("train_loss = ",temp_train_loss)
        if temp_train_loss>train_loss:
           temp_train_loss=train_loss
           print({"epoch":epoch,"loss":train_loss,"ACC":train_acc})
           torch.save(model.state_dict(),f'{model_type}_{loss_type}_FPC.pth')
           loss_patience=0
           
        else:
            if loss_patience>patience:
                break
            else:
                loss_patience+=1


except KeyboardInterrupt:
    print("keyboard interrupt")

# model.load_state_dict(torch.load("/code/model.pth"))
# pred,acc = run_test(model,target_test_loader)
# from sklearn.metrics import confusion_matrix,classification_report
# print(classification_report(pred['y_ture'], pred['y_pre']))





