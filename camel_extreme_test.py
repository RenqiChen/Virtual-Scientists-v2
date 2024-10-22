from camel.agents import SciAgent
from camel.messages import BaseMessage
from camel.models import ModelFactory
from camel.types import ModelPlatformType, ModelType
from tqdm import tqdm

model = ModelFactory.create(
    model_platform=ModelPlatformType.OLLAMA,
    model_type="llama3.1",
    url="http://127.0.0.1:11434/v1",
    model_config_dict={"temperature": 0.4},
)

assistant_sys_msg = BaseMessage.make_assistant_message(
    role_name="user",
    content="You are a good man.",
)

agent_pool = []

for i in range(200000):
    agent = SciAgent(assistant_sys_msg, model=model, token_limit=4096, message_window_size = 1)
    agent_pool.append(agent)

user_msg = BaseMessage.make_user_message(
    role_name="user", content="Say OK"
)

msg = user_msg
for i in tqdm(range(200000)):
    agent_id = agent_pool[i]
    msg = agent_id.step(msg).msg
    print(msg.content)
    print('+'*100)
    msg = BaseMessage.make_user_message(
    role_name="user", content="Say OK")