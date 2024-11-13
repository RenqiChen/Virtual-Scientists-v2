salloc --gres=gpu:4 -p vip_gpu_ailab -A ai4agr
OLLAMA_HOST=127.0.0.1:11434 ./ollama serve &
OLLAMA_HOST=127.0.0.1:11435 ./ollama serve &
OLLAMA_HOST=127.0.0.1:11436 ./ollama serve &
OLLAMA_HOST=127.0.0.1:11437 ./ollama serve &
OLLAMA_HOST=127.0.0.1:11438 ./ollama serve &
OLLAMA_HOST=127.0.0.1:11439 ./ollama serve &
OLLAMA_HOST=127.0.0.1:11440 ./ollama serve &
OLLAMA_HOST=127.0.0.1:11441 ./ollama serve &
OLLAMA_HOST=127.0.0.1:11442 ./ollama serve &
OLLAMA_HOST=127.0.0.1:11443 ./ollama serve &
OLLAMA_HOST=127.0.0.1:11444 ./ollama serve &
OLLAMA_HOST=127.0.0.1:11445 ./ollama serve &
OLLAMA_HOST=127.0.0.1:11446 ./ollama serve &
OLLAMA_HOST=127.0.0.1:11447 ./ollama serve &
OLLAMA_HOST=127.0.0.1:11448 ./ollama serve &
OLLAMA_HOST=127.0.0.1:11449 ./ollama serve &
OLLAMA_HOST=127.0.0.1:11450 ./ollama serve &
OLLAMA_HOST=127.0.0.1:11451 ./ollama serve &
OLLAMA_HOST=127.0.0.1:11452 ./ollama serve &
OLLAMA_HOST=127.0.0.1:11453 ./ollama serve &
OLLAMA_HOST=127.0.0.1:11454 ./ollama serve &
OLLAMA_HOST=127.0.0.1:11455 ./ollama serve &
OLLAMA_HOST=127.0.0.1:11456 ./ollama serve &

sbatch -N 1 --gres=gpu:4 -p vip_gpu_ailab -A ai4agr run.sh
sbatch -N 1 --gres=gpu:4 -p vip_gpu_ailab -A ai4agr run_3.sh

srun -p ai4agr -n 1 -N 1 --gres=gpu:4 hostname

salloc --gpus=4 -N 2 -p vip_gpu_ailab -A ai4agr  --qos=gpugpu


登录节点
tmux new -s name
tmux a -t name
tmux ls
ctrl+b c
ctrl+b w
计算节点
python run.py 2>&1 | tee output.txt


output_small :24000, big port
output_2: origin 100000, small port
output_3: origin 100000, big port
salloc fast: 50000, big port

df -Th /home/bingxing2/ailab/scxlab0066/

OLLAMA_HOST=0.0.0.0:11434 ./ollama serve &
sbatch -N 2 --gres=gpu:4 --qos=gpugpu -p vip_gpu_ailab -A ai4agr port2.sh

### multi-GPU
sbatch -N 1 --gres=gpu:4 -p vip_gpu_ailab -A ai4agr port1.sh

sbatch -N 1 --gres=gpu:4 -p vip_gpu_ailab -A ai4agr port2.sh

sbatch -N 2 --gres=gpu:4 --qos=gpugpu -p vip_gpu_ailab -A ai4agr port2.sh