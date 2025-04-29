import sys
sys.path.append('camel-master')
from camel.agents import SciAgent
from camel.messages import BaseMessage
from camel.models import ModelFactory
from camel.types import ModelPlatformType, ModelType
import sqlite3
import os
import random
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time

def upper_triangle_index_to_position(n, index):
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            if count == index:
                return (i, j)
            count += 1
    raise IndexError("Index out of bounds for upper triangle.")

def select_authors(trainable_adjmatrix, sample_num, total_agent_num, random_sample = False):
    # sampling n agent means sampling n-1 edges
    sample_edge_num = sample_num - 1
    if sample_edge_num > len(trainable_adjmatrix):
        raise ValueError("Cannot sample more elements than available without replacement.")

    # sample ID of trainable_adjmatrix to coordination
    sampleID2coord = {}
    for i in range(len(trainable_adjmatrix)):
        sampleID2coord[i] = upper_triangle_index_to_position(total_agent_num, i)
    # coordination to sample ID of trainable_adjmatrix
    coord2sampleID = {}
    for k, v in sampleID2coord.items():
        coord2sampleID[v] = k

    author_ids = []
    sampled_ids = []
    edge_probs = torch.exp(trainable_adjmatrix.detach().clone()).tolist()
    edge_ids = [i for i in range(len(trainable_adjmatrix))]

    for i in range(sample_edge_num):
        if random_sample:
            # 1. random sample an edge
            sampled_position = random.sample([i for i in range(len(edge_ids))], 1)[0]
            # 2. put authors related to the link into the list
            sampled_edge_id = edge_ids[sampled_position]
            author_ids.extend(sampleID2coord[sampled_edge_id])
        else:
            # 1. random sample an edge
            sampled_position = torch.multinomial(torch.tensor(edge_probs), num_samples=1, replacement=False)[0]  # 1. random sample an edge
            # 2. put authors related to the link into the list
            sampled_edge_id = edge_ids[sampled_position]
            author_ids.extend(sampleID2coord[sampled_edge_id])

        # 3. delete duplicated authors
        author_ids = list(set(author_ids))

        # 4. update edge_ids and edge_probs
        edge_probs.pop(sampled_position)
        edge_ids.pop(sampled_position)
        edge_probs_ = []
        edge_ids_ = []
        for i in range(len(edge_ids)):
            if sampleID2coord[edge_ids[i]][0] in author_ids or sampleID2coord[edge_ids[i]][1] in author_ids:
                edge_probs_.append(edge_probs[i])
                edge_ids_.append(edge_ids[i])
        edge_probs = edge_probs_
        edge_ids = edge_ids_

    # 5. update sampled_ids
    author_ids.sort()
    for i in range(len(author_ids)):
        for j in range(i+1, len(author_ids)):
            sampled_ids.append(coord2sampleID[(i, j)])
    return sampled_ids, author_ids

def generate_prompts(agent_num, output_dir):
    agent_prompts = []
    agent_fields = []
    global_conn = sqlite3.connect(os.path.join('/home/bingxing2/ailab/group/ai4agr/crq/SciSci/OAG/new_OAG_from_2010to2020_gt_100_citation',
                                               'author_database.db'))
    global_cursor = global_conn.cursor()
    # load all authors
    author_list = global_cursor.execute('SELECT * FROM authors').fetchall()[:100]

    # random select agent_num authors
    sampled_ids = random.sample(range(1, len(author_list)), agent_num)

    for random_id in sampled_ids:
        author_info = author_list[random_id]
        prompt = (f'You are Scientist {random_id}, you are from {author_info[2]}, your research interests are {", ".join(author_info[5].split(";")[:5])}. '
                  + f'You are a helpful assistant. Your task is to answer question the given question through discussion with other scientists. Directly say what you want to say.')
        agent_prompts.append(prompt)
        agent_fields.append(author_info[2])
    global_cursor.close()
    global_conn.close()

    # save agent_prompts
    with open(os.path.join(output_dir, 'agent_prompts.txt'), 'w') as f:
        for prompt in agent_prompts:
            f.write(prompt + '\n')
            f.write('==='*10 + '\n')

    return agent_prompts, sampled_ids, agent_fields

def convert_list2matrix(trainable_adjmatrix, agent_num):
    matrix = np.zeros((agent_num, agent_num))
    index = 0
    for i in range(agent_num):
        for j in range(i + 1, agent_num):
            matrix[i][j] = trainable_adjmatrix[index]
            matrix[j][i] = trainable_adjmatrix[index]
            index += 1
    return np.exp(matrix)

def draw_trainable_adjmatrix(trainable_adjmatrix, output_dir):
    # draw trainable_adjmatrix as a heatmap
    plt.figure(figsize=(10, 10))
    sns.heatmap(trainable_adjmatrix, annot=True, fmt=".3f", cmap='coolwarm', square=True)
    plt.title('Trainable Adjacency Matrix')
    plt.xlabel('Agent ID')
    plt.ylabel('Agent ID')
    # Display the plot
    plt.savefig(os.path.join(output_dir, 'Trainable Adjacency Matrix.png'), dpi=300)
    plt.close()

def draw_overall_reward_trend(overall_reward_trend, output_dir):
    # draw overall reward trend
    plt.figure(figsize=(10, 8))
    plt.plot(overall_reward_trend)
    plt.title('Overall Reward Trend')
    plt.xlabel('Epoch')
    plt.ylabel('Overall Reward')
    plt.savefig(os.path.join(output_dir, 'Overall Reward Trend.png'), dpi=300)
    plt.close()

if __name__ == "__main__":
    total_epoch = 1800
    beginning_epoch_id = 0 # if start with a checkpoint then this would not be 0
    max_discussion_turn = 1
    learning_rate = 0.001
    model_type = 'llama3'

    sample_size = 4
    agent_num = 10 # total number of init agents
    epsilon = 0.1

    output_dir = f'RL_Multi_Agent_Test_Result_{sample_size}_out_of_{agent_num}_with_{model_type}'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    device = torch.device("cuda:0")
    trainable_adjmatrix = torch.nn.Parameter(torch.rand(int(agent_num*(agent_num-1)/2)), requires_grad=True)
    trainable_adjmatrix.to(device)
    all_agent_prompts, all_agent_ids, agent_fields = generate_prompts(agent_num, output_dir)
    optimizer = torch.optim.Adam([trainable_adjmatrix], lr=learning_rate)
    overall_reward_trend = []

    # Check if checkpoint exists
    if os.path.exists(os.path.join(output_dir, 'checkpoint.pth')):
        checkpoint = torch.load(os.path.join(output_dir, 'checkpoint.pth'), map_location=device)
        with torch.no_grad():
            trainable_adjmatrix.copy_(checkpoint['trainable_adjmatrix'].to(device))
        trainable_adjmatrix.requires_grad_(True)  # <== ensure it keeps requires_grad=True

        optimizer = torch.optim.Adam([trainable_adjmatrix], lr=learning_rate)  # re-create optimizer with new parameter
        optimizer.load_state_dict(checkpoint['optimizer'])
        overall_reward_trend = checkpoint['overall_reward_trend']
        print(f"Checkpoint loaded")

    model = ModelFactory.create(
        model_platform=ModelPlatformType.OLLAMA,
        model_type=model_type,
        embed_model_type = "",
        url="http://127.0.0.1:11434/v1",
        model_config_dict={"temperature": 0.4},
    )

    # record time
    start_time = time.time()
    for epoch in range(len(overall_reward_trend), total_epoch):
        print(f'epoch {epoch} start')
        random_number = random.random()
        print(f'random_number {random_number}')
        agent_team = []
        # random select team_size agents
        if random_number < epsilon or epoch == 0:
            sampled_ids, author_ids = select_authors(trainable_adjmatrix, sample_size, agent_num, random_sample = True)
        else:
            sampled_ids, author_ids = select_authors(trainable_adjmatrix, sample_size, agent_num, random_sample=False)

        # init agents
        for author_id in author_ids:
            # format prompt
            format_prompt = BaseMessage.make_assistant_message(
                role_name=f"Scientist{all_agent_ids[author_id]}",
                content=all_agent_prompts[author_id])
            agent_team.append(SciAgent(format_prompt, model=model, token_limit=4096, message_window_size=1))

        # discussion
        questions = ['You go at red, but stop at green. What am I?',
                     'I’m tall when I’m young, and I’m short when I’m old. What am I?',
                     'I have a head and a tail that will never meet. Having too many of me is always a treat. What am I?',
                     'I help you from your head to your toe. The more I work, the smaller I grow. What am I?',
                     'I can fly but have no wings. I can cry but I have no eyes. Wherever I go, darkness follows me. What am I?'
                     'I’m (usually) white and used for cutting and grinding. When I’m damaged, humans usually remove me or fill me. For most animals I am a useful tool. What am I?',
                     'I shave every day, but my beard stays the same. What am I?',
                     'I’m where yesterday follows today and tomorrow is in the middle. What am I?',
                     'I’m a god, a planet, and I measure heat. What am I?',
                     'I have branches, but no fruit, trunk or leaves. What am I?']

        answers = ['watermelon',
                   'candle',
                   'coin',
                   'soap',
                   'cloud',
                   'tooth',
                   'barber',
                   'dictionary'
                   'mercury',
                   'bank']
        reward_list = []
        for question, answer in zip(questions, answers):
            global_prompt = (
                f'You are in a team consists of {", ".join([f"Scientist {all_agent_ids[id]} from {agent_fields[id]}" for id in author_ids])} from different fields. You need to collaborate with each other and answer the following riddle: '
                f'{question} '
                f'**Reply \"BINGO\" if you think you are ready to answer.**')
            print(global_prompt)
            print('+' * 100)

            for i in range(max_discussion_turn):
                memory_list = []
                end = False
                for agent in agent_team:
                    # keep memory size less than len(agent_team)
                    if len(memory_list) > len(agent_team):
                        memory_list.pop()
                    # init user message
                    msg = BaseMessage.make_user_message(
                        role_name="user",
                        content=global_prompt + '\t' + '\t'.join(memory_list)
                    )
                    # agent reply
                    msg = agent.step(msg)
                    print(agent.role_name + ': ' + msg.msg.content)
                    print('+' * 100)
                    memory_list.append(agent.role_name + ': ' + msg.msg.content)

                    # check if all agents are ready to answer
                    if 'bingo' in msg.msg.content.lower():
                        end = True
                        break

                if end or i==max_discussion_turn-1:
                    memory_list.append('user: Tell me the answer.')
                    msg = BaseMessage.make_user_message(
                        role_name="user",
                        content=global_prompt + '\t' + '\t'.join(memory_list)
                    )
                    msg = agent.step(msg)
                    print(agent.role_name + ': ' + msg.msg.content)
                    print('+' * 100)
                    if answer in msg.msg.content.lower():
                        reward = 1
                    else:
                        reward = -1
                    break

            # update trainable_adjmatrix
            loss = -reward * torch.pow(torch.exp(trainable_adjmatrix[sampled_ids]).prod(), 1/len(sampled_ids))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            reward_list.append(-loss.item())

        # update overall reward trend
        overall_reward_trend.append(sum(reward_list) / len(reward_list))
        # draw overall reward trend
        draw_overall_reward_trend(overall_reward_trend, output_dir)
        # reform trainable_adjmatrix to a symmetric matrix
        trainable_adjmatrix4fig = convert_list2matrix(trainable_adjmatrix, agent_num)
        draw_trainable_adjmatrix(trainable_adjmatrix4fig, output_dir)
        # Save checkpoint
        checkpoint = {
            'trainable_adjmatrix': trainable_adjmatrix.detach().cpu(),
            'optimizer': optimizer.state_dict(),
            'overall_reward_trend': overall_reward_trend
        }
        torch.save(checkpoint, os.path.join(output_dir, 'checkpoint.pth'))

    # record end time
    end_time = time.time()
    # Calculate and print running time
    elapsed_time = end_time - start_time
    print(f"Total running time: {elapsed_time/60:.2f} minutes")