#!/bin/sh
module load compilers/cuda/12.1
module load cudnn/8.8.1.3_cuda12.x
module load compilers/gcc/12.2.0
source activate mamba_ssm_cp311

cd /home/bingxing2/ailab/scxlab0066/SocialScience/SciSci/ollama
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

cd /home/bingxing2/ailab/scxlab0066/SocialScience/Social_Science_CAMEL/sci_platform
python run_fast.py 2>&1 | tee output_fast_5.txt