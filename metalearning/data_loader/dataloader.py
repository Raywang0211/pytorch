import numpy as np
import torch.utils.data as tud



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