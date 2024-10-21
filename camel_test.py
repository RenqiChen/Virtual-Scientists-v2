from camel.agents import SciAgent
from camel.messages import BaseMessage
from camel.models import ModelFactory
from camel.types import ModelPlatformType, ModelType

model = ModelFactory.create(
    model_platform=ModelPlatformType.OLLAMA,
    model_type="llama3.1",
    url="http://127.0.0.1:11434/v1",
    model_config_dict={"temperature": 0.4},
)

# model = ModelFactory.create(
#     model_platform=ModelPlatformType.INTERN,
#     model_type = ModelType.INTERN_VL, # enum type class, value for model name
#     api_key="sk-vgtwqiedawdmdzckjxlfkrurxovcitprueethwihzilszbku",
#     url="https://api.siliconflow.cn/v1",
#     model_config_dict={"temperature": 0.4},
# )

assistant_sys_msg0 = BaseMessage.make_assistant_message(
    role_name="Scientist0",
    content="You are Scientist0, You are a helpful assistant. Your task is to answer questions from Scientist1. Directly say what you want to say.",
)
agent0 = SciAgent(assistant_sys_msg0, model=model, token_limit=4096, message_window_size = 1)


assistant_sys_msg1 = BaseMessage.make_assistant_message(
    role_name="Scientist1",
    content="You are Scientist1, you have questions about protein to ask Scientist0. Directly say what you want to say.",
)
agent1 = SciAgent(assistant_sys_msg1, model=model, token_limit=4096, message_window_size = 1)

user_msg = BaseMessage.make_user_message(
    role_name="user", content=""
)

msg = user_msg
for i in range(1):
    msg = agent0.step(msg)
    print(msg.msg.content)
    print('+'*100)
    msg = BaseMessage.make_assistant_message(
        role_name=agent0.role_name,
        content=msg.msg.content,
    )

    msg = agent1.step(msg)
    print(msg.msg.content)
    print('+'*100)
    msg = BaseMessage.make_assistant_message(
        role_name=agent1.role_name,
        content=msg.msg.content,
    )