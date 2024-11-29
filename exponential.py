import numpy as np
import matplotlib.pyplot as plt

# 参数
lambda_val = 0.25  # 指数分布的参数
scale = 1 / lambda_val  # scale是1/λ

# 生成样本，假设生成1000个数据点
sample_size = 1000
samples = np.random.exponential(scale, sample_size)+3
# 取整
samples = np.floor(samples).astype(int)
# set the range between 3 and 10
samples = np.clip(samples, 3, 11)
# 随机从samples中采样一个值
print(samples)
# 可视化生成的分布
plt.hist(samples, bins=30, density=True, alpha=0.6, color='g')

# 画出理论的概率密度函数
x = np.linspace(0, np.max(samples), 1000)
y = lambda_val * np.exp(-lambda_val * (x-3))
plt.plot(x, y, 'r-', lw=2)

plt.title(r'Histogram of samples and $e^{-0.26x}$ PDF')
plt.xlabel('x')
plt.ylabel('Density')
plt.show()
# save
save_path = '/home/bingxing2/ailab/scxlab0066/SocialScience/exponential.png'
plt.savefig(save_path)