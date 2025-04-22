# dirs
root_dir = "/home/bingxing2/ailab/group/ai4agr/crq/SciSci/OAG"
ollama_dir = "/home/bingxing2/ailab/suhaoyang/soft/new-ollama/bin"

# deploy setups
use_ollama = True
ips = ['paraai-n32-h-01-agent-112','paraai-n32-h-01-agent-161','paraai-n32-h-01-agent-162','paraai-n32-h-01-agent-173','paraai-n32-h-01-agent-208','paraai-n32-h-01-agent-218','127.0.0.1']
port = list(range(11434, 11450))
port4GPU = 4

# exp setups
agent_num = 3000
runs = 1
team_limit = 3
max_discuss_iteration = 1
max_team_member = 6
epochs = 50
checkpoint = True
leader_mode = 'normal' 
test_time = '0422_3000'
load_time = '0421_2_3000'