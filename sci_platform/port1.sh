#!/bin/sh
module load compilers/cuda/12.1
module load cudnn/8.8.1.3_cuda12.x
module load compilers/gcc/12.2.0
source activate mamba_ssm_cp311

cd /home/bingxing2/ailab/scxlab0066/SocialScience/SciSci/ollama
OLLAMA_HOST=0.0.0.0:11434 ./ollama serve &
OLLAMA_HOST=0.0.0.0:11435 ./ollama serve &
OLLAMA_HOST=0.0.0.0:11436 ./ollama serve &
OLLAMA_HOST=0.0.0.0:11437 ./ollama serve &
OLLAMA_HOST=0.0.0.0:11438 ./ollama serve &
OLLAMA_HOST=0.0.0.0:11439 ./ollama serve &
OLLAMA_HOST=0.0.0.0:11440 ./ollama serve &
OLLAMA_HOST=0.0.0.0:11441 ./ollama serve &
OLLAMA_HOST=0.0.0.0:11442 ./ollama serve &
OLLAMA_HOST=0.0.0.0:11443 ./ollama serve &
OLLAMA_HOST=0.0.0.0:11444 ./ollama serve &
OLLAMA_HOST=0.0.0.0:11445 ./ollama serve &
OLLAMA_HOST=0.0.0.0:11446 ./ollama serve &
OLLAMA_HOST=0.0.0.0:11447 ./ollama serve &
OLLAMA_HOST=0.0.0.0:11448 ./ollama serve &
OLLAMA_HOST=0.0.0.0:11449 ./ollama serve 
