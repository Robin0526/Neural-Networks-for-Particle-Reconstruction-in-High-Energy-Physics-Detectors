import numpy as np
import glob
import os
import re
import tifffile as tiff
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dropout, UpSampling2D, concatenate, ZeroPadding2D, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
import tensorflow as tf
import albumentations as A

# 指定数据文件夹路径
folder_path = 'C:/Users/25610/Desktop/online 2/Gauss_S1.00_NL0.30_B0.01'
# 指定保存路径
save_path = 'C:/Users/25610/Desktop/online 2/save'
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
scaler_X = MinMaxScaler()  # 使用 Min-Max 归一化
scaler_Y = MinMaxScaler()  # 使用 Min-Max 归一化

# 不要扁平化Y数据，直接处理其原有的形状
X_scaled = scaler_X.fit_transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)
Y_scaled = scaler_Y.fit_transform(Y.reshape(-1, Y.shape[-1]))

# 重塑Y_scaled回原始形状
Y_scaled = Y_scaled.reshape(Y.shape)

# 划分训练集、验证集和测试集
X_train, X_temp, Y_train, Y_temp = train_test_split(X_scaled, Y_scaled, test_size=0.3, random_state=42)
X_val, X_test, Y_val, Y_test = train_test_split(X_temp, Y_temp, test_size=0.33, random_state=42)

# 数据增强
transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0, p=0.5),
    A.Rotate(limit=30, p=0.5),
    A.RandomCrop(height=48, width=48, p=0.5),
    A.Resize(height=56, width=56, p=1)
])

# 数据增强函数
def augment_data(x, y):
    augmented = transform(image=x.astype(np.float32), mask=y.astype(np.float32))
    x_aug = augmented['image']
    y_aug = augmented['mask']
    return x_aug, y_aug

# 创建 tf.data.Dataset
def create_dataset(X, Y, batch_size=32):
    dataset = tf.data.Dataset.from_tensor_slices((X, Y))
    dataset = dataset.shuffle(buffer_size=len(X))
    dataset = dataset.map(lambda x, y: tf.numpy_function(augment_data, [x, y], [tf.float32, tf.float32]), num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.map(lambda x, y: (tf.ensure_shape(x, X.shape[1:]), tf.ensure_shape(y, Y.shape[1:])), num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return dataset

train_dataset = create_dataset(X_train, Y_train)
val_dataset = create_dataset(X_val, Y_val)

# 定义U-Net模型
def unet_model(input_size=(56, 56, 4)):
    inputs = Input(input_size)
    
    # 编码器部分
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    conv1 = BatchNormalization()(conv1)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
    conv1 = BatchNormalization()(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = BatchNormalization()(conv2)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
    conv2 = BatchNormalization()(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = BatchNormalization()(conv3)
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
    conv3 = BatchNormalization()(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    
    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool3)
    conv4 = BatchNormalization()(conv4)
    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv4)
    conv4 = BatchNormalization()(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)
    
    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool4)
    conv5 = BatchNormalization()(conv5)
    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv5)
    conv5 = BatchNormalization()(conv5)
    drop5 = Dropout(0.5)(conv5)
    
    # 解码器部分
    up6 = Conv2D(256, (2, 2), activation='relu', padding='same')(UpSampling2D(size=(2, 2))(drop5))
    up6 = ZeroPadding2D(padding=((0, 1), (0, 1)))(up6)  # 调整形状以匹配 drop4
    merge6 = concatenate([drop4, up6], axis=3)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(merge6)
    conv6 = BatchNormalization()(conv6)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv6)
    conv6 = BatchNormalization()(conv6)
    
    up7 = Conv2D(128, (2, 2), activation='relu', padding='same')(UpSampling2D(size=(2, 2))(conv6))
    merge7 = concatenate([conv3, up7], axis=3)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(merge7)
    conv7 = BatchNormalization()(conv7)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv7)
    conv7 = BatchNormalization()(conv7)
    
    up8 = Conv2D(64, (2, 2), activation='relu', padding='same')(UpSampling2D(size=(2, 2))(conv7))
    merge8 = concatenate([conv2, up8], axis=3)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(merge8)
    conv8 = BatchNormalization()(conv8)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv8)
    conv8 = BatchNormalization()(conv8)
    
    up9 = Conv2D(32, (2, 2), activation='relu', padding='same')(UpSampling2D(size=(2, 2))(conv8))
    merge9 = concatenate([conv1, up9], axis=3)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(merge9)
    conv9 = BatchNormalization()(conv9)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv9)
    conv9 = BatchNormalization()(conv9)
    
    conv10 = Conv2D(1, (1, 1), activation='linear')(conv9)
    
    model = Model(inputs=inputs, outputs=conv10)
    return model

# 创建U-Net模型
model = unet_model(input_size=(56, 56, 4))

# 编译模型
model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

# 打印模型结构，检查输出形状是否正确
model.summary()

# 定义回调函数
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.0001)
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# 训练模型
model.fit(train_dataset, 
          epochs=100, 
          validation_data=val_dataset, 
          callbacks=[reduce_lr, early_stopping])

# 使用测试集进行测试
y_pred = model.predict(X_test)

# 保存预测结果和真实值
np.save(os.path.join(save_path, 'predictions.npy'), y_pred)
np.save(os.path.join(save_path, 'truth.npy'), Y_test)