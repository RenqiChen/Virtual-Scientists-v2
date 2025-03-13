# dirs
root_dir = "/home/bingxing2/ailab/group/ai4agr/crq/SciSci/OAG"
ollama_dir = "/home/bingxing2/ailab/suhaoyang/soft/new-ollama/bin"

# deploy setups
use_ollama = True
ips = ['paraai-n32-h-01-agent-118','paraai-n32-h-01-agent-127','paraai-n32-h-01-agent-131','paraai-n32-h-01-agent-136','paraai-n32-h-01-agent-171','paraai-n32-h-01-agent-174','paraai-n32-h-01-agent-181','paraai-n32-h-01-agent-216','127.0.0.1']
port = list(range(11434, 11450))
port4GPU = 4

# exp setups
agent_num = 3000
runs = 1
team_limit = 3
max_discuss_iteration = 1
max_team_member = 6
epochs = 60
checkpoint = True
test_time = '0313_3000'
load_time = '0312_3000'