#!/bin/sh
module load compilers/cuda/12.1
module load cudnn/8.8.1.3_cuda12.x
module load compilers/gcc/12.2.0
source activate mamba_ssm_cp311

cd /home/bingxing2/ailab/scxlab0066/SocialScience/SciSci/ollama
CUDA_VISIBLE_DEVICES=0 OLLAMA_HOST=0.0.0.0:11446 ./ollama serve &
CUDA_VISIBLE_DEVICES=0 OLLAMA_HOST=0.0.0.0:11447 ./ollama serve &
CUDA_VISIBLE_DEVICES=0 OLLAMA_HOST=0.0.0.0:11448 ./ollama serve &
CUDA_VISIBLE_DEVICES=1 OLLAMA_HOST=0.0.0.0:11449 ./ollama serve &
CUDA_VISIBLE_DEVICES=1 OLLAMA_HOST=0.0.0.0:11450 ./ollama serve &
CUDA_VISIBLE_DEVICES=1 OLLAMA_HOST=0.0.0.0:11451 ./ollama serve &
CUDA_VISIBLE_DEVICES=2 OLLAMA_HOST=0.0.0.0:11452 ./ollama serve &
CUDA_VISIBLE_DEVICES=2 OLLAMA_HOST=0.0.0.0:11453 ./ollama serve &
CUDA_VISIBLE_DEVICES=2 OLLAMA_HOST=0.0.0.0:11454 ./ollama serve &
CUDA_VISIBLE_DEVICES=3 OLLAMA_HOST=0.0.0.0:11455 ./ollama serve &
CUDA_VISIBLE_DEVICES=3 OLLAMA_HOST=0.0.0.0:11456 ./ollama serve &
CUDA_VISIBLE_DEVICES=3 OLLAMA_HOST=0.0.0.0:11457 ./ollama serve &

cd /home/bingxing2/ailab/scxlab0066/SocialScience/Social_Science_CAMEL/sci_platform
python run_fast.py 2>&1 | tee output_large.txt