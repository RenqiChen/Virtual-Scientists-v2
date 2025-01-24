from tqdm import tqdm
from time import sleep
import psutil
from datetime import datetime

# 打开文件以追加模式写入
with open('usage_log.txt', 'a') as logfile:
    with tqdm(total=100, desc='cpu%', position=1) as cpubar, tqdm(total=100, desc='ram%', position=0) as rambar:
        while True:
            ram_percent = psutil.virtual_memory().percent
            cpu_percent = psutil.cpu_percent()
            current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            # 更新进度条
            rambar.n = ram_percent
            cpubar.n = cpu_percent
            rambar.refresh()
            cpubar.refresh()
            
            # 记录到文件并刷新缓冲区
            logfile.write(f'Time: {current_time}, RAM: {ram_percent}%, CPU: {cpu_percent}%\n')
            logfile.flush()
            
            sleep(5)