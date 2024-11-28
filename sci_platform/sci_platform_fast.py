import sys
import os
import numpy as np
import json
import re
import ollama
from functools import partial
import faiss
from typing import Any

sys.path.append('../camel-master')
from camel.agents import SciAgent
from camel.messages import BaseMessage
from camel.models import ModelFactory
from camel.types import ModelPlatformType, ModelType
from social_agent.channel import Channel
from social_agent.sci_agent import SciAgent_Async

from sci_team.SciTeam import Team
from utils.prompt import Prompts
from utils.scientist_utils import (
    team_description,
    convert_you_to_other,
    team_description_detail,
    read_txt_files_as_dict,
    extract_between_json_tags,
    count_team,
    save2database,
    read_txt_files_as_list,
    process_author_text
)

import asyncio
from inference.inference_manager import InferencerManager

class Platform:
    r"""Platform."""

    def __init__(self,
                 model_configuration: str = './configs/model_configs.json',
                 agent_num: int = 1,
                 ips: list = ['127.0.0.1'],
                 port: list = [11434],
                 #  root_dir: str = '/home/bingxing2/ailab/group/ai4agr/shy/s4s',
                 root_dir: str = '/home/bingxing2/ailab/group/ai4agr/crq/SciSci/OAG',

                 #  author_folder_path: str = "/home/bingxing2/ailab/group/ai4agr/crq/SciSci/books",
                 author_folder_path: str = 'books_OAG_3169_after',
                 # /home/bingxing2/ailab/group/ai4agr/crq/SciSci/OAG/books_OAG_3169_after

                 #  paper_folder_path: str = "/home/bingxing2/ailab/group/ai4agr/crq/SciSci/papers",
                 paper_folder_path: str = 'papers_OAG',
                 # /home/bingxing2/ailab/group/ai4agr/crq/SciSci/OAG/papers_OAG

                 #  future_paper_folder_path: str = "/home/bingxing2/ailab/group/ai4agr/crq/SciSci/papers_future",
                 future_paper_folder_path: str = 'papers_future_OAG',
                 # /home/bingxing2/ailab/group/ai4agr/crq/SciSci/OAG/papers_future_OAG

                 author_info_dir: str = 'authors',
                 #  adjacency_matrix_dir: str = 'authors_degree_ge50_from_year2000to2010',
                 adjacency_matrix_dir: str = 'data_from_2010to2020_gt_200_citation/3169_authors_w_15_degrees_15_papers',
                 agent_model_config_name: str = 'ollama_llama3.1_8b',
                 review_model_config_name: str = 'ollama_llama3.1_70b',
                 knowledgeBank_config_dir: str = './configs/knowledge_config.json',
                 log_dir: str = 'logs',
                 info_dir: str = "team_info",
                 group_max_discuss_iteration: int = 2, # 6， 7
                 recent_n_team_mem_for_retrieve: int = 3,
                 recent_n_agent_mem_for_retrieve: int = 1,
                 team_limit: int = 2,
                 check_iter: int = 5,
                 review_num: int = 2,
                 max_teammember: int = 3,
                 cite_number: int = 8,
                 default_mark: int = 4,
                 skip_check: bool = False,
                 over_state: int = 7,
                 begin_state: int = 1,
                 inference_configs: dict[str, Any] | None = None,
                 explore: str = 'gaussian', # 'uniform' or 'gaussian'
                 ):

        author_folder_path = os.path.join(root_dir, author_folder_path)
        paper_folder_path = os.path.join(root_dir, paper_folder_path)
        future_paper_folder_path = os.path.join(root_dir, future_paper_folder_path)
        adjacency_matrix_dir = os.path.join(root_dir, adjacency_matrix_dir)

        self.agent_num = agent_num
        self.port = port
        self.ips = ips
        self.paper_folder_path = paper_folder_path
        self.paper_future_folder_path = future_paper_folder_path
        self.author_info_dir = os.path.join(author_folder_path,'author_{}.txt')
        self.adjacency_matrix_dir = adjacency_matrix_dir
        self.group_max_discuss_iteration = group_max_discuss_iteration
        self.recent_n_team_mem_for_retrieve = recent_n_team_mem_for_retrieve
        self.recent_n_agent_mem_for_retrieve = recent_n_agent_mem_for_retrieve
        # how many teams for one agent is allowed
        self.team_limit = team_limit
        # how many times to try paper search
        self.check_iter = check_iter
        # the number of reviewer
        self.reviewer_num = review_num
        # the max team member in a team
        self.max_teammember = max_teammember
        # cite how many paper when generating the idea
        self.cite_number = cite_number
        # default review mark
        self.default_mark = default_mark
        # check novelty
        self.skip_check = skip_check
        # current state for the over of team activity
        self.over_state = over_state
        # current state for the begin of team activity
        self.begin_state = begin_state
        # output dir
        self.log_dir = log_dir
        self.info_dir = info_dir
        self.author_folder_path = author_folder_path
        self.explore = explore

        # for quality, the team of one member will think more times
        self.think_times = max_teammember+1

        # author2paper file: dict{'authorID':[paperID1, paperID2, ...]}
        # with open('{}/author2paper.json'.format(root_dir), 'r') as file:
        #     self.author2paper = json.load(file)

        # load k-hop adjacency matrix
        self.degree_int2word = ['one', 'two', 'three', 'four', 'five']
        # self.adjacency_matrix = np.loadtxt(
        #     '{}/{}-hop_adj_matrix.txt'.format(self.adjacency_matrix_dir, self.degree_int2word[hop_num-1]), dtype=int)
        self.adjacency_matrix = np.loadtxt(
            '{}/weight_matrix.txt'.format(self.adjacency_matrix_dir), dtype=int)

        # check if agent_num is valid
        if self.agent_num is None:
            self.agent_num = len(self.adjacency_matrix)
        else:
            assert self.agent_num <= len(self.adjacency_matrix)

        # load agentID2authorID file: dict{'agentID': 'authorID'}
        # with open('{}/agentID2authorID.json'.format(self.adjacency_matrix_dir), 'r') as file:
        #     self.agentID2authorID = json.load(file)

        # init model
        model = ModelFactory.create(
            model_platform=ModelPlatformType.OLLAMA,
            model_type="llama3.1",
            embed_model_type = "mxbai-embed-large",
            url="http://127.0.0.1:11434/v1",
            model_config_dict={"temperature": 0.4},
        )

        self.inference_channel = Channel()
        self.embed_inference_channel = Channel()
        self.inference_channel_reviewer = Channel()
        self.embed_inference_channel_reviewer = Channel()
        self.inference_configs = {
            'model_type': "llama3.1",
            'embed_model_type': None,
            'model_path': 'API',
            'stop_tokens': None,
            'server_url': [
                {
                'host': 'paraai-n32-h-01-agent-173',
                'ports': self.port[:12]
                },
                {
                'host': 'paraai-n32-h-01-agent-148',
                'ports': self.port[:12]
                },
                {
                'host': 'paraai-n32-h-01-agent-171',
                'ports': self.port[:12]
                },
                {
                'host': 'paraai-n32-h-01-agent-65',
                'ports': self.port[:12]
                },
                {
                'host': 'paraai-n32-h-01-agent-112',
                'ports': self.port[:12]
                },
                {
                'host': '127.0.0.1',
                'ports': self.port[12:]
                },
          ]
        }
        self.embed_inference_configs = {
            'model_type': 'llama3.1',
            'embed_model_type': "mxbai-embed-large",
            'model_path': 'API',
            'stop_tokens': None,
            'server_url': [
                {
                'host': 'paraai-n32-h-01-agent-173',
                'ports': self.port[:12]
                },
                {
                'host': 'paraai-n32-h-01-agent-148',
                'ports': self.port[:12]
                },
                {
                'host': 'paraai-n32-h-01-agent-171',
                'ports': self.port[:12]
                },
                {
                'host': 'paraai-n32-h-01-agent-65',
                'ports': self.port[:12]
                },
                {
                'host': 'paraai-n32-h-01-agent-112',
                'ports': self.port[:12]
                },
                {
                'host': '127.0.0.1',
                'ports': self.port[12:]
                },
          ]
        }
        self.infere = InferencerManager(
            self.inference_channel,
            **self.inference_configs,
        )
        self.embed_infere = InferencerManager(
            self.embed_inference_channel,
            **self.embed_inference_configs,
        )
        self.infere_reviewer = InferencerManager(
            self.inference_channel_reviewer,
            **self.inference_configs,
        )
        self.embed_infere_reviewer = InferencerManager(
            self.embed_inference_channel_reviewer,
            **self.embed_inference_configs,
        )

        # model = ModelFactory.create(
        #     model_platform=ModelPlatformType.INTERN,
        #     model_type = ModelType.INTERN_VL,
        #     embed_model_type = None,
        #     api_key="sk-gaqeyxvrpmjondtmihuydxjevoztjgibkfmfqyhgmonhfsml",
        #     url="https://api.siliconflow.cn/v1",
        #     model_config_dict={"temperature": 0.4},
        # )

        # init agent pool
        # self.agent_pool = [self.init_agent(str(agent_id), model, '/home/bingxing2/ailab/group/ai4agr/crq/SciSci/books/author_{}.txt'.format(agent_id)) for agent_id in range(len(self.adjacency_matrix))]
        self.agent_pool = self.init_agent_async(model, self.inference_channel, self.embed_inference_channel, self.author_info_dir, len(self.adjacency_matrix))
        # self.reviewer_pool = [self.init_reviewer(str(agent_id), model) for agent_id in range(self.reviewer_num)]
        self.reviewer_pool = self.init_reviewer_async(model, self.inference_channel_reviewer, self.embed_inference_channel_reviewer, self.reviewer_num)
        self.id2agent = {}
        for agent in self.agent_pool:
            self.id2agent[agent.role_name] = agent
        # team pool
        self.team_pool = []
        agent_id = 1
        for agent in self.agent_pool[:self.agent_num]:
            team_agent = []
            team_index = []
            team_index.append(agent.role_name)
            team_dic = Team(team_name = str(agent_id)+','+str(1),
                            log_dir = self.log_dir,
                            info_dir = self.info_dir,
                            recent_n_team_mem_for_retrieve = self.recent_n_team_mem_for_retrieve)
            team_dic.teammate = team_index
            team_agent.append(team_dic)
            self.team_pool.append(team_agent)
            agent_id = agent_id + 1

        # paper embedding list
        # cpu_index = faiss.read_index("/home/bingxing2/ailab/group/ai4agr/crq/SciSci/faiss_index.index")  # 加载索引
        cpu_index = faiss.read_index(os.path.join(root_dir, 'faiss_index_OAG.index'))
        # /home/bingxing2/ailab/group/ai4agr/crq/SciSci/OAG/faiss_index_OAG.index

        res = faiss.StandardGpuResources()  # 为 GPU 资源分配
        self.gpu_index = faiss.index_cpu_to_gpu(res, 0, cpu_index)  # 将索引移到 GPU

        cpu_authors_index = faiss.read_index(os.path.join(root_dir, 'faiss_index_authors.index'))  # 加载索引

        authors_res = faiss.StandardGpuResources()  # 为 GPU 资源分配
        self.gpu_authors_index = faiss.index_cpu_to_gpu(authors_res, 0, cpu_authors_index)  # 将索引移到 GPU

        # cpu_future_index = faiss.read_index("/home/bingxing2/ailab/group/ai4agr/crq/SciSci/faiss_index_future.index")  # 加载索引
        cpu_future_index = faiss.read_index(os.path.join(root_dir, 'faiss_index_OAG_future.index'))
        # /home/bingxing2/ailab/group/ai4agr/crq/SciSci/OAG/faiss_index_OAG_future.index

        future_res = faiss.StandardGpuResources()  # 为 GPU 资源分配
        self.gpu_future_index = faiss.index_cpu_to_gpu(future_res, 0, cpu_future_index)  # 将索引移到 GPU

        self.paper_dicts = read_txt_files_as_dict(self.paper_folder_path)
        # self.author_dicts = read_txt_files_as_list(self.author_folder_path)
        self.paper_future_dicts = read_txt_files_as_dict(self.paper_future_folder_path)

    def init_reviewer(self, agent_id, model):
        name = 'Paper Reviewer{}'.format(agent_id)
        prompt = BaseMessage.make_assistant_message(
            role_name=name,
            content=f'You are {name}. ' + Prompts.prompt_review_system,
        )
        agent = SciAgent(prompt, model=model, token_limit=4096, message_window_size = self.recent_n_agent_mem_for_retrieve)
        return agent

    def init_reviewer_async(self, model, channel, embed_channel, count):
        agents=[]
        inference_channel=channel
        for i in range(count):
            name = 'Paper Reviewer{}'.format(i)
            prompt = BaseMessage.make_assistant_message(
                role_name=name,
                content=f'You are {name}. ' + Prompts.prompt_review_system,
            )
            agent = SciAgent_Async(prompt, model=model, channel=inference_channel, embed_channel=embed_channel, token_limit=4096)
            agents.append(agent)
        return agents

    def init_agent(self, agent_id, model, information_path):
        # load author info
        with open(information_path, 'r') as file:
            prompt = file.read()
        name = 'Scientist{}'.format(agent_id)
        prompt = BaseMessage.make_assistant_message(
            role_name=name,
            content=prompt,
        )
        agent = SciAgent(prompt, model=model, token_limit=4096, message_window_size = self.recent_n_agent_mem_for_retrieve)

        return agent

    def init_agent_async(self, model, channel, embed_channel, information_path, count):
        agents = []
        inference_channel = channel

        # 创建并注册用户
        for i in range(count):
            information_path_index = information_path.format(i)
            with open(information_path_index, 'r') as file:
                prompt = file.read()
            name = 'Scientist{}'.format(i)
            prompt = BaseMessage.make_assistant_message(
                role_name=name,
                content=prompt,
            )
            agent = SciAgent_Async(prompt, model=model, channel=inference_channel, embed_channel=embed_channel, token_limit=4096, message_window_size = self.recent_n_agent_mem_for_retrieve)
            agents.append(agent)

        return agents

    async def select_single(self, agent_index, explore='uniform'):
        scientists = self.agent_pool[:self.agent_num]
        # avoid too many teams
        if count_team(self.team_pool[agent_index], self.over_state)>=self.team_limit:
            return
        sys_prompt = scientists[agent_index].orig_sys_message.content + Prompts.role
        hint = BaseMessage.make_user_message(role_name="user",
                                             content=Prompts.ask_choice.format_map(
                                                 {"Scientist_name": scientists[agent_index].role_name,
                                                  "All_team": team_description(self.team_pool[agent_index],
                                                                               self.over_state)})
                                             )
        x = await scientists[agent_index].step(hint)
        x= x.msg
        self.team_pool[agent_index][0].log_dialogue('user', hint.content)
        self.team_pool[agent_index][0].log_dialogue(scientists[agent_index].role_name, x.content)
        match = re.search(r'(\d+)', extract_between_json_tags(x.content), re.IGNORECASE)
        if match != None:
            # when action2, the agent choose to act independently
            if int(match.group(1))==2:
                print("Single Agent Independently!")
                self.team_pool[agent_index][0].state=2
                return

        # use prompts to select scientists
        scientist = scientists[agent_index].role_name
        name = int(scientist[9:])
        arr = self.adjacency_matrix[name, :].copy()
        # uniform distribution
        if explore == 'uniform':
            arr += 1
        # sample from gaussian distribution
        else:
            random_values = np.random.normal(loc=1, scale=1, size=self.adjacency_matrix.shape)
            random_values = np.abs(random_values)
            random_values[random_values > 10] = 10
            arr += random_values
        arr[agent_index] = 0
        probabilities = arr / np.sum(arr)

        # team member follows the distribution
        team_sample = np.random.normal(loc=self.max_teammember, scale=1)
        team_sample_int = int(np.round(team_sample))
        team_size = np.clip(team_sample_int, 3, self.max_teammember+2)
        
        selected_indices = np.random.choice(len(arr), size=team_size, p=probabilities, replace=False)

        team_candidate = []
        for i in range(len(selected_indices)):
            team_candidate.append(f"Scientist{selected_indices[i]}")

        # print(team_candidate)
        self.team_pool[agent_index][0].log_dialogue(scientists[agent_index].role_name, ','.join(team_candidate))

        # ask each scientist to decide whether to join
        agent_candidate = self.id_to_agent(team_candidate)
        # create new team
        team_index = []
        team_index.append(scientists[agent_index].role_name)
        for agent in agent_candidate:
            if agent.role_name == scientists[agent_index].role_name:
                continue

            hint = BaseMessage.make_user_message(content=Prompts.to_scientist_choice.format_map({
                "inviter_name": scientists[agent_index].role_name,
                "team_member": str(team_index),
                "personal information" : convert_you_to_other(sys_prompt)
            }), role_name="User")
            # set_parsers(agent, Prompts.scientist_invite_parser)
            pattern = re.compile(r'1', re.IGNORECASE)
            # action1 means a scientist accepts the invitance
            x = await agent.step(hint)
            x = x.msg
            if pattern.search(extract_between_json_tags(x.content, num=1)):
                team_index.append(agent.role_name)
            # self.team_pool[agent_index][0].log_dialogue('user', hint.content)
            # self.team_pool[agent_index][0].log_dialogue(agent.role_name, x.content)

        team_dic = Team(team_name = str(agent_index+1)+','+str(len(self.team_pool[agent_index])+1),
                        log_dir = self.log_dir,
                        info_dir = self.info_dir,
                        recent_n_team_mem_for_retrieve = self.recent_n_team_mem_for_retrieve)
        team_dic.state=2
        team_dic.teammate = team_index
        self.team_pool[agent_index].append(team_dic)

        # connetion between collaborators will be closer
        for member in team_dic.teammate:
            if int(member[9:])!=agent_index:
                self.adjacency_matrix[agent_index, int(member[9:])]=self.adjacency_matrix[agent_index, int(member[9:])]+0.2
                self.adjacency_matrix[int(member[9:]), agent_index]=self.adjacency_matrix[int(member[9:]), agent_index]+0.2
        # summary current teams in memory
        summary_select = await scientists[agent_index].step(BaseMessage.make_user_message(
            content=team_description_detail(self.team_pool[agent_index], self.agent_pool, self.over_state),
            role_name="User"))
        self.team_pool[agent_index][0].log_dialogue(scientists[agent_index].role_name, summary_select.msg.content)

    async def select_coauthors(self,explore='uniform'):
        scientists = self.agent_pool[:self.agent_num]
        # decide whether the scientist wants to find partners
        select_tasks = []
        for agent_index in range(len(scientists)):
            select_tasks.append(self.select_single(agent_index, explore=explore))
        await asyncio.gather(*select_tasks)  # 并行执行所有任务
        team_list = self.team_pool
        return team_list

    def id_to_agent(self, team_list):
        agent_list = []
        for agent_id in team_list:
            agent_list.append(self.id2agent[agent_id])
        return agent_list

    def agent_to_id(self, team_list):
        agent_list = []
        for agent_id in team_list:
            agent_list.append(agent_id.role_name)
        return agent_list

    def reference_paper(self, query_vector, cite_number, epoch):
        D, I = self.gpu_index.search(query_vector, cite_number)

        paper_use = []
        for id in range(len(I[0])):
            if epoch<=self.paper_dicts[I[0][id]]['year']:
                continue
            paper_title = self.paper_dicts[I[0][id]]['title']
            paper_abstract = self.paper_dicts[I[0][id]]['abstract']
            paper_index = {}
            paper_index['title'] = paper_title
            paper_index['abstract'] = paper_abstract
            paper_use.append(paper_index)
        paper_reference = ""
        for id in range(len(paper_use)):
            paper_index = paper_use[id]
            paper_reference = paper_reference+"Paper {}:".format(id+1)+"\n"
            paper_reference = paper_reference+"Title: "+paper_index['title']+"\n"
            paper_reference = paper_reference+"Abstract: "+paper_index['abstract']+"}"+"\n"
        return paper_reference, I[0]

    def reference_author(self, key_string, cite_number):
        query_vector = ollama.embeddings(model="mxbai-embed-large", prompt=key_string)
        query_vector = np.array([query_vector['embedding']])
        D, I = self.gpu_authors_index.search(query_vector, cite_number)

        author_use = []
        for id in range(len(I[0])):
            author = self.author_dicts[I[0][id]]
            author_index = process_author_text(author)
            author_use.append(author_index)
        author_reference = ""
        for id in range(len(author_use)):
            author_index = author_use[id]
            author_reference = author_reference+author_index+"\n"
        return author_reference

    async def team_running(self, epoch, leader_index):
        leader_team=[]
        for team_index in range(len(self.team_pool[leader_index])):
            self.team_pool[leader_index][team_index].epoch = epoch
            await self.team_pool[leader_index][team_index].action_excution(self)
            if self.team_pool[leader_index][team_index].state != self.over_state:
                leader_team.append(self.team_pool[leader_index][team_index])
            # if self.team_pool[leader_index][team_index].state == self.over_state:
            #     self.team_pool[leader_index][team_index].save_team_info()
        self.team_pool[leader_index] = leader_team

    async def running(self, epochs):
        # 创建调度任务
        self.inference_task = asyncio.create_task(self.infere.run())
        self.embed_inference_task = asyncio.create_task(self.embed_infere.run())
        self.inference_task_reviewer = asyncio.create_task(self.infere_reviewer.run())
        self.embed_inference_task_reviewer = asyncio.create_task(self.embed_infere_reviewer.run())
        # init team_pool
        print(f'{"="*50}Epoch:{-1} | Initialize Teams {"="*50}')
        self.team_pool = await self.select_coauthors(self.explore)

        for epoch in range(epochs):
            # state 7 is an over
            # 1. select coauthors for state 1
            # 2. select topics for state 2
            # 3. generate idea for state 3
            # 4. check novelty for state 4
            # 5. generate paper abstract for state 5
            # 6. generate paper review for state 6
            leader_tasks = []
            for leader_index in range(len(self.team_pool)):
                leader_tasks.append(self.team_running(epoch, leader_index))

            await asyncio.gather(*leader_tasks)  # 并行执行所有任务

            print(f'{"="*50} Epoch:{epoch} | Begin Select Authors {"="*50}')
            self.team_pool = await self.select_coauthors(self.explore)
            print(f'{"="*50} Epoch:{epoch} | Current Action Finished {"="*50}')

        await self.infere.stop()
        await self.embed_infere.stop()
        await self.infere_reviewer.stop()
        await self.embed_infere_reviewer.stop()
        # 等待task.run完成，防止主程序结束kill子线程(即inference_task)
        await self.inference_task,self.inference_task_reviewer
        await self.embed_inference_task,self.embed_inference_task_reviewer
        output_dir = "/home/bingxing2/ailab/scxlab0066/SocialScience/database/database_large.db"
        save2database(self.paper_dicts, output_dir)
        # save self.adjacency_matrix
        np.savetxt('/home/bingxing2/ailab/scxlab0066/SocialScience/database/weight_matrix.txt', self.adjacency_matrix, fmt='%d')
    