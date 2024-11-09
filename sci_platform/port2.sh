#!/bin/sh
module load compilers/cuda/12.1
module load cudnn/8.8.1.3_cuda12.x
module load compilers/gcc/12.2.0
source activate mamba_ssm_cp311

cd /home/bingxing2/ailab/scxlab0066/SocialScience/SciSci/ollama
OLLAMA_HOST=0.0.0.0:11450 ./ollama serve &
OLLAMA_HOST=0.0.0.0:11451 ./ollama serve &
OLLAMA_HOST=0.0.0.0:11452 ./ollama serve &
OLLAMA_HOST=0.0.0.0:11453 ./ollama serve &
OLLAMA_HOST=0.0.0.0:11454 ./ollama serve &
OLLAMA_HOST=0.0.0.0:11455 ./ollama serve &
OLLAMA_HOST=0.0.0.0:11456 ./ollama serve &
OLLAMA_HOST=0.0.0.0:11457 ./ollama serve &
OLLAMA_HOST=0.0.0.0:11458 ./ollama serve &
OLLAMA_HOST=0.0.0.0:11459 ./ollama serve &
OLLAMA_HOST=0.0.0.0:11460 ./ollama serve &
OLLAMA_HOST=0.0.0.0:11461 ./ollama serve &
OLLAMA_HOST=0.0.0.0:11462 ./ollama serve &
OLLAMA_HOST=0.0.0.0:11463 ./ollama serve &
OLLAMA_HOST=0.0.0.0:11464 ./ollama serve &
OLLAMA_HOST=0.0.0.0:11465 ./ollama serve &

cd /home/bingxing2/ailab/scxlab0066/SocialScience/Social_Science_CAMEL/sci_platform
python run_fast.py 2>&1 | tee output_large.txt