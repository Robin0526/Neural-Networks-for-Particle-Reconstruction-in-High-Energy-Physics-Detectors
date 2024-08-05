import numpy as np
import glob
import os
import re
import tifffile as tiff
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, UpSampling2D
from tensorflow.keras.optimizers import Adam
from keras.models import load_model



# 指定数据文件夹路径
folder_path = "/hy-tmp/Gauss_S1.00_NL0.30_B0.50"
# 指定保存路径
save_path = '/hy-tmp/Results_3*3'
# 确保目录存在
os.makedirs(save_path, exist_ok=True)

# 定义自然排序函数
def natural_sort_key(s):
    sub_strings = re.split(r'(\d+)', s)
    sub_strings = [int(c) if c.isdigit() else c for c in sub_strings]
    return sub_strings

# 获取不同类别图片路径并加入到对应的列表中
emcal_list = sorted(glob.glob(os.path.join(folder_path, 'emcal_*')), key=natural_sort_key)
hcal_list = sorted(glob.glob(os.path.join(folder_path, 'hcal_*')), key=natural_sort_key)
trkn_list = sorted(glob.glob(os.path.join(folder_path, 'trkn_*')), key=natural_sort_key)
trkp_list = sorted(glob.glob(os.path.join(folder_path, 'trkp_*')), key=natural_sort_key)
truth_list = sorted(glob.glob(os.path.join(folder_path, 'truth_*')), key=natural_sort_key)

# 读取图像数据
def load_images(file_list):
    images = [tiff.imread(p) for p in file_list]
    return np.array(images)

emcal_data = load_images(emcal_list)
hcal_data = load_images(hcal_list)
trkn_data = load_images(trkn_list)
trkp_data = load_images(trkp_list)
truth_data = load_images(truth_list)

# 合并前四张图作为输入，第五张图作为输出
X = np.stack([emcal_data, hcal_data, trkn_data, trkp_data], axis=-1)
Y = truth_data

# 数据归一化
scaler_X = StandardScaler()
scaler_Y = StandardScaler()

# 注意：这里我们不需要扁平化Y数据，因为我们将直接处理其原有的形状
X_scaled = scaler_X.fit_transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)
Y_scaled = scaler_Y.fit_transform(Y.reshape(-1, Y.shape[-1]))

# 重塑Y_scaled回原始形状
Y_scaled = Y_scaled.reshape(Y.shape)

# 划分训练集、验证集和测试集
X_train, X_temp, Y_train, Y_temp = train_test_split(X_scaled, Y_scaled, test_size=0.3, random_state=42)
X_val, X_test, Y_val, Y_test = train_test_split(X_temp, Y_temp, test_size=0.33, random_state=42)

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(56, 56, 4)),
    Conv2D(32, (3, 3), activation='relu', padding='same'),  # 新增一层
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu', padding='same'),
    Conv2D(64, (3, 3), activation='relu', padding='same'),  # 新增一层
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu', padding='same'),
    Conv2D(128, (3, 3), activation='relu', padding='same'),
    Conv2D(128, (3, 3), activation='relu', padding='same'),  # 新增一层
    UpSampling2D(size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu', padding='same'),
    Conv2D(64, (3, 3), activation='relu', padding='same'),  # 新增一层
    UpSampling2D(size=(2, 2)),
    Conv2D(32, (3, 3), activation='relu', padding='same'),  # 新增一层，以保持尺寸
    Conv2D(1, (1, 1), activation='linear', padding='same')  
])

# 编译模型
model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

# 打印模型结构，检查输出形状是否正确
model.summary()

# 训练模型
model.fit(X_train, Y_train, epochs=100, batch_size=32, validation_data=(X_val, Y_val))


#model.save('my_model1.keras')  

# 使用测试集进行测试
y_pred = model.predict(X_test)

# 保存预测结果和真实值
np.save(os.path.join(save_path, 'predictions.npy'), y_pred)
np.save(os.path.join(save_path, 'truth.npy'), Y_test)


