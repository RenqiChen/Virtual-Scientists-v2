# deploy setups
ips = ['paraai-n32-h-01-agent-112','paraai-n32-h-01-agent-126','paraai-n32-h-01-agent-161','paraai-n32-h-01-agent-162','paraai-n32-h-01-agent-173','paraai-n32-h-01-agent-218','127.0.0.1']
port = list(range(11434, 11450))

# exp setups
agent_num = 2000
runs = 1
team_limit = 3
max_discuss_iteration = 1
max_team_member = 6
epochs = 30
model_name = "llama3.1"
leader_mode = 'normal' 

# checkpoint setups
checkpoint = False
test_time = '0424_3000'
load_time = '0424_3000'