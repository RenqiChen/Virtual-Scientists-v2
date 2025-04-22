### multi-GPU

sbatch -N 1 --gres=gpu:4 -p vip_gpu_ailab -A ai4agr port1.sh

sbatch -N 1 --gres=gpu:4 -p vip_gpu_ailab -A ai4agr port2.sh
