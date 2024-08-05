import numpy as np
import matplotlib.pyplot as plt
import os

# 指定保存路径
save_path = '/hy-tmp/Results_3*3'

# 加载预测结果和真实值
y_pred = np.load(os.path.join(save_path, 'predictions.npy'))
Y_test = np.load(os.path.join(save_path, 'truth.npy'))

# 确保保存目录存在
os.makedirs(save_path, exist_ok=True)

# 可视化函数
def visualize_results(predictions, truths, save_path, num_samples=3):
    for i in range(num_samples):
        plt.figure(figsize=(12, 6))

        # 显示预测图像
        plt.subplot(1, 2, 1)
        plt.imshow(predictions[i], cmap='gray')
        plt.title(f'Prediction {i+1}')
        plt.axis('off')

        # 显示真实图像
        plt.subplot(1, 2, 2)
        plt.imshow(truths[i], cmap='gray')
        plt.title(f'Truth {i+1}')
        plt.axis('off')

        # 保存图像
        plt.savefig(os.path.join(save_path, f'comparison_{i+1}.png'))
        plt.show()  # 显示图像
        plt.close()

# 调用可视化函数
visualize_results(y_pred, Y_test, save_path)
