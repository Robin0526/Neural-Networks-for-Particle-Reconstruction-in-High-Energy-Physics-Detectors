# %%
import torch
import torch.nn as nn
from torch.utils.data import DataLoader,Dataset
from torchvision import transforms
from torchvision import datasets
import matplotlib.pyplot as plt


# %%
import numpy as np
import skimage.io as io
import tifffile as tiff
import glob
import os
import re
import matplotlib.pyplot as plt
import pandas as pd


# %% [markdown]
# # 读取数据

# %%
epochs=100

model_name='/hy-tmp/cnn/cnn_B0.50_100epoch.pth'
pred_data_name='/hy-tmp/cnn/100pt_predictions.npy'
true_data_name='/hy-tmp/cnn/100pt_truth.npy'
folder_path = "Gauss_S1.00_NL0.30_B0.50"
plot_path='/hy-tmp/cnn/tv_100loss.png'
x_scale='log'

# %%
#文件名自然排序
def natural_sort_key(s):
    sub_strings = re.split(r'(\d+)', s)
    sub_strings = [int(c) if c.isdigit() else c for c in sub_strings]
    return sub_strings

# %%

#获取不同类别图片路径并加入到对应的list中
emcal_list = sorted(glob.glob(os.path.join(folder_path, 'emcal_*')), key=natural_sort_key)
hcal_list = sorted(glob.glob(os.path.join(folder_path, 'hcal_*')), key=natural_sort_key)
trkn_list=sorted(glob.glob(os.path.join(folder_path, 'trkn_*')), key=natural_sort_key)
trkp_list=sorted(glob.glob(os.path.join(folder_path, 'trkp_*')), key=natural_sort_key)
truth_list = sorted(glob.glob(os.path.join(folder_path, 'truth_*')), key=natural_sort_key)

specified_files = list(zip(emcal_list, hcal_list, trkn_list,trkp_list, truth_list))




emcal_data,hcal_data,trkn_data,trkp_data,truth_data=np.zeros((10000,56,56)),np.zeros((10000,56,56)),np.zeros((10000,56,56)),np.zeros((10000,56,56)),np.zeros((10000,56,56))
for i in range(len(emcal_list)):
    emcal_data[i]=io.imread(specified_files[i][0])
    hcal_data[i]=io.imread(specified_files[i][1])
    trkn_data[i]=io.imread(specified_files[i][2])
    trkp_data[i]=io.imread(specified_files[i][3])
    truth_data[i]=io.imread(specified_files[i][4])


# %%
from sklearn.model_selection import train_test_split

# %%
# 创建 PyTorch 数据集
class CustomDataset(Dataset):
    def __init__(self, emcal_data, hcal_data, trkn_data, trkp_data, truth_data, transform=None):
        self.emcal_data = emcal_data
        self.hcal_data = hcal_data
        self.trkn_data = trkn_data
        self.trkp_data = trkp_data
        self.truth_data = truth_data
        self.transform = transform

    def __len__(self):
        return len(self.emcal_data)

    def __getitem__(self, idx):  
        x_emcal = torch.tensor(self.emcal_data[idx], dtype=torch.float32)  
        x_hcal = torch.tensor(self.hcal_data[idx], dtype=torch.float32)  
        x_trkn = torch.tensor(self.trkn_data[idx], dtype=torch.float32)  
        x_trkp = torch.tensor(self.trkp_data[idx], dtype=torch.float32)  
        y_truth = torch.tensor(self.truth_data[idx], dtype=torch.float32)  
        
        # 堆叠为多通道输入，原shape为[4, 56, 56]  
        x = torch.stack([x_emcal, x_hcal, x_trkn, x_trkp], dim=0)  # 变成 [4, 56, 56]  

        # 添加一个维度使其变为 [1, 4, 56, 56]  
        y_truth = y_truth.unsqueeze(0)  # 在维度 0 处添加一个维度  

        if self.transform:  
            x = self.transform(x)  
            y_truth = self.transform(y_truth)  

        return x, y_truth  

# 拆分数据集
train_emcal, rem_emcal, train_hcal, rem_hcal, train_trkn, rem_trkn, train_trkp, rem_trkp, train_truth, rem_truth = train_test_split(
    emcal_data, hcal_data, trkn_data, trkp_data, truth_data, test_size=0.3, random_state=42
)
test_emcal, val_emcal, test_hcal, val_hcal, test_trkn, val_trkn, test_trkp, val_trkp, test_truth, val_truth = train_test_split(
    rem_emcal, rem_hcal,rem_trkn, rem_trkp, rem_truth, test_size=0.5, random_state=42
)


# 创建训练和验证数据集
train_dataset = CustomDataset(train_emcal, train_hcal, train_trkn, train_trkp, train_truth)
test_dataset = CustomDataset(test_emcal, test_hcal, test_trkn, test_trkp, test_truth)
val_dataset = CustomDataset(val_emcal, val_hcal,val_trkn, val_trkp, val_truth)

# 创建 DataLoader
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)




class Inception(nn.Module):
    def __init__(self, in_channels):
        super(Inception, self).__init__()
        self.branch1x1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=1),
            nn.BatchNorm2d(32),
            nn.ReLU(True)
        )
        
        self.branch3x3 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True)
        )
        
        self.branch5x5 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Conv2d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(True)
        )
        
        self.branch_pool = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, 32, kernel_size=1),
            nn.BatchNorm2d(32),
            nn.ReLU(True)
        )
        
    def forward(self, x):
        branch1 = self.branch1x1(x)
        
        branch3 = self.branch3x3(x)
        
        branch5 = self.branch5x5(x)
        
        branch_pool = self.branch_pool(x)
        
        outputs = [branch1, branch3, branch5, branch_pool]
        return torch.cat(outputs, 1)

class CNN(nn.Module):  
    def __init__(self):
        super(CNN, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            
            Inception(in_channels=128),  # Add Inception Block here           
            nn.Conv2d(in_channels=192, out_channels=128, kernel_size=1),  # Adjust out_channels to match InceptionBlock output
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            
               
            
            Inception(in_channels=128),  # Add Inception Block here           
            nn.Conv2d(in_channels=192, out_channels=128, kernel_size=1),  # Adjust out_channels to match InceptionBlock output
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            
#             Inception(in_channels=128),  # Add Inception Block here           
#             nn.Conv2d(in_channels=192, out_channels=64, kernel_size=1),  # Adjust out_channels to match InceptionBlock output
#             nn.BatchNorm2d(64),
#             nn.ReLU(True),
            
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            
            
            
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            
            nn.Conv2d(in_channels=32, out_channels=1, kernel_size=3, padding=1)
        )


    def forward(self, x):  
        y = self.net(x)  
        return y  # 输出形状为 [32, 1, 56, 56]


# %%
# 创建子类的实例，并搬到GPU上
model = CNN().to('cuda:0')

# %%
# 损失函数的选择
loss_fn = nn.MSELoss()

# %%
# 优化算法的选择

learning_rate = 0.005    # 设置学习率  
weight_decay = 0.001    # 设置权重衰减  

optimizer = torch.optim.Adam(  
    model.parameters(),   
    lr=learning_rate,   
    weight_decay=weight_decay  # 添加权重衰减  
)  

# %%
import torch  
import matplotlib.pyplot as plt  
 


batch_interval = 500  # 每2000个小批次打印一次损失
train_losses = [] 
val_losses = [] 

for epoch in range(epochs):
    print(f'Epoch {epoch + 1}/{epochs}')
    model.train()  # 设置模型为训练模式
    for batch_idx, (x, y) in enumerate(train_loader):
        x, y = x.to('cuda'), y.to('cuda')
        
        # 前向传播
        Pred = model(x)
        loss = loss_fn(Pred, y)
        

        # 每 batch_interval 打印一次训练损失
        if batch_idx % batch_interval == 0:
            print(f'Batch {batch_idx}, Train Loss: {loss.item()}')
        
        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    train_losses.append(loss.item()) 
    # 计算验证集损失
    model.eval()  # 设置模型为评估模式
    val_loss = 0
    with torch.no_grad():
        for x_val, y_val in val_loader:
            x_val, y_val = x_val.to('cuda'), y_val.to('cuda')
            Pred_val = model(x_val)
            val_loss += loss_fn(Pred_val, y_val).item()
    
    val_loss /= len(val_loader)
    val_losses.append(val_loss)
    print(f'Epoch {epoch + 1}, Validation Loss: {val_loss}')

# 绘制训练损失和验证损失  
Fig = plt.figure()  
plt.plot(range(len(train_losses)), train_losses, label='Train Loss')  
plt.plot(range(len(val_losses)), val_losses, label='Validation Loss')  
if x_scale == 'log':  
    plt.yscale('log')  

# 添加x和y标签  
plt.xlabel('Epochs')          # 替换为适当的x标签  
plt.ylabel('Loss')            # 替换为适当的y标签  

plt.legend()  
plt.savefig(plot_path)  
plt.show()  

# 保存模型
torch.save(model.state_dict(), model_name)
print(model_name)


# 初始化评估指标
total_mse = 0
total_mae = 0
total_samples = 0
all_predictions = []
all_targets = []

# 测试网络
with torch.no_grad():  # 该局部关闭梯度计算功能
    for (x, y) in test_loader:  # 获取小批次的 x 与 y
        x, y = x.to('cuda'), y.to('cuda')
        Pred = model(x)  # 一次前向传播（小批量）
        
        # 累加 MSE 和 MAE
        total_mse += torch.sum((Pred - y) ** 2).item()
        total_mae += torch.sum(torch.abs(Pred - y)).item()
        total_samples += y.size(0)
        
        # 收集所有预测值和真实值
        all_predictions.append(Pred.cpu().numpy())
        all_targets.append(y.cpu().numpy())

# 计算最终的 MSE 和 MAE
mse = total_mse / total_samples
mae = total_mae / total_samples


all_predictions = np.concatenate(all_predictions, axis=0)
all_targets = np.concatenate(all_targets, axis=0)




# 保存预测结果和真实值
np.save( pred_data_name,all_predictions)
np.save( true_data_name, all_targets)