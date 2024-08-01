import numpy as np
import os
import tifffile as tiff
import matplotlib.pyplot as plt
import glob
import re
import skimage.io as io
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam

def natural_sort_key(s):
    sub_strings = re.split(r'(\d+)', s)
    sub_strings = [int(c) if c.isdigit() else c for c in sub_strings]
    return sub_strings

folder_path = "E:/Gauss_1.0_0.3/Gauss_1.0_0.3"

emcal_list = sorted(glob.glob(os.path.join(folder_path, 'emcal_*')), key=natural_sort_key)
hcal_list = sorted(glob.glob(os.path.join(folder_path, 'hcal_*')), key=natural_sort_key)
tracker_list = sorted(glob.glob(os.path.join(folder_path, 'tracker_*')), key=natural_sort_key)
truth_list = sorted(glob.glob(os.path.join(folder_path, 'truth_*')), key=natural_sort_key)

specified_files = list(zip(emcal_list, hcal_list, tracker_list, truth_list))

# 分离出测试集
test_files = specified_files[-3:]
train_files = specified_files[:-3]

# 准备训练数据和验证数据
emcal_train_data, hcal_train_data, tracker_train_data, truth_train_data = [], [], [], []
emcal_val_data, hcal_val_data, tracker_val_data, truth_val_data = [], [], [], []

for emcal_path, hcal_path, tracker_path, truth_path in train_files:
    emcal_train_data.append(io.imread(emcal_path))
    hcal_train_data.append(io.imread(hcal_path))
    tracker_train_data.append(io.imread(tracker_path))
    truth_train_data.append(io.imread(truth_path))

for emcal_path, hcal_path, tracker_path, truth_path in test_files:
    emcal_val_data.append(io.imread(emcal_path))
    hcal_val_data.append(io.imread(hcal_path))
    tracker_val_data.append(io.imread(tracker_path))
    truth_val_data.append(io.imread(truth_path))

# 转换数据格式
emcal_train_data = np.array(emcal_train_data).reshape(-1, 56, 56, 1)
hcal_train_data = np.array(hcal_train_data).reshape(-1, 56, 56, 1)
tracker_train_data = np.array(tracker_train_data).reshape(-1, 56, 56, 1)
truth_train_data = np.array(truth_train_data).reshape(-1, 56, 56, 1)

emcal_val_data = np.array(emcal_val_data).reshape(-1, 56, 56, 1)
hcal_val_data = np.array(hcal_val_data).reshape(-1, 56, 56, 1)
tracker_val_data = np.array(tracker_val_data).reshape(-1, 56, 56, 1)
truth_val_data = np.array(truth_val_data).reshape(-1, 56, 56, 1)

# 合并输入数据
train_input_data = np.concatenate([emcal_train_data, hcal_train_data, tracker_train_data], axis=-1)
val_input_data = np.concatenate([emcal_val_data, hcal_val_data, tracker_val_data], axis=-1)

# 构建更深更复杂的模型
model = Sequential([
    Conv2D(64, (3, 3), activation='relu', input_shape=(56, 56, 3)),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Conv2D(256, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(56 * 56, activation='sigmoid')
])

model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')

# 训练模型
model.fit(train_input_data, truth_train_data.reshape(-1, 56 * 56), epochs=100, batch_size=32, validation_data=(val_input_data, truth_val_data.reshape(-1, 56 * 56)))

# 测试模型
for idx in range(len(test_files)):
    test_input = val_input_data[idx:idx+1]
    predicted_truth = model.predict(test_input).reshape(56, 56)
    true_truth = truth_val_data[idx].reshape(56, 56)
    
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 4, 1)
    plt.title('EMCAL')
    plt.imshow(emcal_val_data[idx].reshape(56, 56), cmap='gray')
    plt.subplot(1, 4, 2)
    plt.title('HCAL')
    plt.imshow(hcal_val_data[idx].reshape(56, 56), cmap='gray')
    plt.subplot(1, 4, 3)
    plt.title('Tracker')
    plt.imshow(tracker_val_data[idx].reshape(56, 56), cmap='gray')
    plt.subplot(1, 4, 4)
    plt.title('Predicted Truth')
    plt.imshow(predicted_truth, cmap='gray')
    plt.show()
    
    plt.figure(figsize=(5, 5))
    plt.title('True Truth')
    plt.imshow(true_truth, cmap='gray')
    plt.show()
