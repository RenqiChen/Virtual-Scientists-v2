#!/bin/sh
module load compilers/cuda/12.1
module load cudnn/8.8.1.3_cuda12.x
module load compilers/gcc/12.2.0
source activate mamba_ssm_cp311

cd /home/bingxing2/ailab/scxlab0066/SocialScience/SciSci/ollama
CUDA_VISIBLE_DEVICES=0 OLLAMA_HOST=0.0.0.0:11434 ./ollama serve &
CUDA_VISIBLE_DEVICES=0 OLLAMA_HOST=0.0.0.0:11435 ./ollama serve &
CUDA_VISIBLE_DEVICES=1 OLLAMA_HOST=0.0.0.0:11436 ./ollama serve &
CUDA_VISIBLE_DEVICES=1 OLLAMA_HOST=0.0.0.0:11437 ./ollama serve &
CUDA_VISIBLE_DEVICES=1 OLLAMA_HOST=0.0.0.0:11438 ./ollama serve &
CUDA_VISIBLE_DEVICES=2 OLLAMA_HOST=0.0.0.0:11439 ./ollama serve &
CUDA_VISIBLE_DEVICES=2 OLLAMA_HOST=0.0.0.0:11440 ./ollama serve &
CUDA_VISIBLE_DEVICES=2 OLLAMA_HOST=0.0.0.0:11441 ./ollama serve &
CUDA_VISIBLE_DEVICES=3 OLLAMA_HOST=0.0.0.0:11442 ./ollama serve &
CUDA_VISIBLE_DEVICES=3 OLLAMA_HOST=0.0.0.0:11443 ./ollama serve &
CUDA_VISIBLE_DEVICES=3 OLLAMA_HOST=0.0.0.0:11444 ./ollama serve &

cd /home/bingxing2/ailab/scxlab0066/SocialScience/Social_Science_CAMEL/sci_platform
python run_fast.py 2>&1 | tee output_large.txt