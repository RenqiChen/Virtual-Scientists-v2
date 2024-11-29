# dirs
root_dir = "/home/bingxing2/ailab/group/ai4agr/crq/SciSci/OAG"
ollama_dir = "/home/bingxing2/ailab/suhaoyang/soft/new-ollama/bin"

# deploy setups
use_ollama = True
ips = ['paraai-n32-h-01-agent-173','paraai-n32-h-01-agent-171','paraai-n32-h-01-agent-65','paraai-n32-h-01-agent-148','paraai-n32-h-01-agent-112','127.0.0.1']
port = list(range(11434, 11446))
port4GPU = 3

# exp setups
agent_num = 10
runs = 1
team_limit = 2
max_discuss_iteration = 1
max_team_member = 3
epochs = 50