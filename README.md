salloc --gres=gpu:2 -p vip_gpu_ailab -A ai4agr
OLLAMA_HOST=127.0.0.1:11434 ./ollama serve
OLLAMA_HOST=127.0.0.1:11435 ./ollama serve
OLLAMA_HOST=127.0.0.1:11436 ./ollama serve

OLLAMA_HOST=127.0.0.1:11437 ./ollama serve