
import numpy as np
import skimage.io as io
import tifffile as tiff
import glob
import os
import re

import matplotlib.pyplot as plt

from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.neural_network import MLPClassifier 
from sklearn.preprocessing import StandardScaler


#文件名自然排序
def natural_sort_key(s):
    sub_strings = re.split(r'(\d+)', s)
    sub_strings = [int(c) if c.isdigit() else c for c in sub_strings]
    return sub_strings


folder_path = "D:/project/dataset/Gauss_S1.00_NL0.30_B0.00"
#folder_path = "D:/project/dataset/Gauss_S1.00_NL0.30_B0.00.zip"

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

def extract_pixel_data(emcal_images, hcal_images, trkn_images,trkp_images,  truth_images):
    num_images, height, width = emcal_images.shape
    X = []
    y = []
    for i in range(num_images):
        for row in range(height):
            for col in range(width):
                features = np.hstack([
                    emcal_images[i, row, col],
                    hcal_images[i, row, col],
                    trkn_images[i, row, col],
                    trkp_images[i, row, col]
                    
                ])
                X.append(features)
                y.append(truth_images[i, row, col])
    return np.array(X), np.array(y)

# 提取特征和标签
X, y = extract_pixel_data(emcal_data, hcal_data, trkn_data,trkp_data, truth_data)

# %%
# 首先，将数据分为训练集和临时集（这里的80%可以根据需要调整）  
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)  # 70% training, 30% temp  

# 然后，再将临时集分为验证集和测试集  
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)  # 50% of temp for validation, 50% for testing  


# %%
# 数据标准化
scaler_X = StandardScaler()
scaler_Y = StandardScaler()

X_train = scaler_X.fit_transform(X_train)
X_test = scaler_X.transform(X_test)
X_val = scaler_X.transform(X_val)
X=scaler_X.transform(X)

y_train = scaler_Y.fit_transform(y_train.reshape(-1,1))
y_test = scaler_Y.transform(y_test.reshape(-1,1))
y_val= scaler_Y.transform(y_val.reshape(-1,1))
y=scaler_Y.transform(y.reshape(-1,1))

mlp = MLPRegressor(hidden_layer_sizes=(256, 128,64), activation='relu', solver='adam', max_iter=5)

model = mlp.fit(X_train, y_train)

# from joblib import dump, load
# dump(model, 'mlp_50000_model_5epoch.joblib')

y_pred=model.predict(X_test)

plt.scatter(y_pred,y_test, s=0.3)
plt.title('Truth vs Predicted Pixel Values (Train Images)')
plt.xlabel('Truth Pixel Value')
plt.ylabel('Predicted Pixel Value')
plt.plot([0, 1], [0, 1], '--', color='gray')
plt.grid()
plt.show()


from sklearn.metrics import mean_squared_error
# 计算均方误差
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse:.4f}")



