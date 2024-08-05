import numpy as np
import os
import matplotlib.pyplot as plt

# 指定保存路径
save_path = '/hy-tmp/Results_3*3'

# 加载保存的预测结果和真实值
y_pred = np.load(os.path.join(save_path, 'predictions.npy'))
Y_test = np.load(os.path.join(save_path, 'truth.npy'))

# 展开为像素点
y_pred_flat = y_pred.flatten()
Y_test_flat = Y_test.flatten()

# 计算所有pred_i - true_i的平均值和标准差
mean_error = np.mean(y_pred_flat - Y_test_flat)
std_dev = np.std(y_pred_flat - Y_test_flat)

# 省略真实值绝对值小于1e-4的像素
mask = np.abs(Y_test_flat) >= 1e-4
pred_i_filtered = y_pred_flat[mask]
true_i_filtered = Y_test_flat[mask]

# 绘制第一个图
fig1, ax1 = plt.subplots(figsize=(8, 4))
ax1.scatter(true_i_filtered, pred_i_filtered - true_i_filtered, s=1, alpha=0.5)
ax1.set_xlabel('True Value')
ax1.set_ylabel('Predicted - True Pixel Value')
ax1.set_title('Predicted - True Pixel Value vs True Value')
ax1.grid(True)
fig1.savefig(os.path.join(save_path, 'plot1.png'))

# 绘制第二个图（x轴刻度从-2到2，间隔为0.25）
fig2, ax2 = plt.subplots(figsize=(8, 4))
n, bins, patches = ax2.hist(pred_i_filtered - true_i_filtered, bins=150, color='skyblue', edgecolor='black')
ax2.set_xlabel('Predicted - True Pixel Value')
ax2.set_ylabel('Frequency')
ax2.set_title('Histogram of Predicted - True Pixel Value')
ax2.set_xlim(-2, 2)
ax2.set_xticks(np.arange(-2, 2.25, 0.25))
ax2.grid(True)
fig2.savefig(os.path.join(save_path, 'plot2.png'))

# 绘制第三个图（线性坐标直方图，x轴刻度从-1到1，间隔为0.25）
fig3, ax3 = plt.subplots(figsize=(8, 4))
n, bins, patches = ax3.hist(pred_i_filtered - true_i_filtered, bins=150, color='skyblue', edgecolor='black')
ax3.set_xlabel('Predicted - True Pixel Value')
ax3.set_ylabel('Frequency')
ax3.set_title('Predicted - True (1D)')
ax3.set_xlim(-1, 1)
ax3.set_xticks(np.arange(-1, 1.25, 0.25))
ax3.grid(True)

# 添加多行文本框，调整位置
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
text1 = f"Mean Error: {mean_error:.4f}\nStandard Deviation: {std_dev:.4f}"
ax3.text(0.05, 0.95, text1, transform=ax3.transAxes,
         verticalalignment='top', horizontalalignment='left', bbox=props, fontsize=10)

# 添加右上角文本框，无背景框，放在图的右上角但在矩形框外面
text2 = "dataset:Gauss_S1.00_NL0.30_B0.50  model:CNN"
ax3.text(1.0, 0.95, text2, transform=ax3.transAxes,
         verticalalignment='top', horizontalalignment='right', fontsize=10)
fig3.savefig(os.path.join(save_path, 'plot3.png'))

# 绘制第四个图（二维直方图）
fig4, ax4 = plt.subplots(figsize=(8, 8))
hb = ax4.hexbin(true_i_filtered, pred_i_filtered, gridsize=50, cmap='Blues', mincnt=1)
cb = fig4.colorbar(hb, ax=ax4, orientation='vertical')
cb.set_label('Counts')
ax4.set_xlabel('True Pixel Value')
ax4.set_ylabel('Predicted Pixel Value')
ax4.set_title('Predicted vs True Pixel Value (2D Histogram)')
ax4.grid(True)

# 添加多行文本框，调整位置
ax4.text(0.05, 0.95, text1, transform=ax4.transAxes,
         verticalalignment='top', horizontalalignment='left', bbox=props, fontsize=10)

# 添加右上角文本框，无背景框，放在图的右上角但在矩形框外面
ax4.text(1.0, 0.98, text2, transform=ax4.transAxes,
         verticalalignment='top', horizontalalignment='right', fontsize=10)
fig4.savefig(os.path.join(save_path, 'plot4.png'))

plt.tight_layout()
plt.show()

# 打印误差统计量
print(f"Mean Error: {mean_error:.4f}")
print(f"Standard Deviation: {std_dev:.4f}")

