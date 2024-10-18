# -*- coding: utf-8 -*-
"""utils."""

from loguru import logger
from tqdm import tqdm
from collections import Counter
from typing import Union, Any, Sequence
import numpy as np
import json
import sqlite3
import re
import os
import sys

sys.path.append('../camel-master')
# from agentscope.service import (
#     dblp_search_publications,  # or google_search,
#     arxiv_search
# )

from utils.prompt import Prompts

def majority_vote(votes: list) -> Any:
    """majority_vote function"""
    votes_valid = [item for item in votes if item != "Abstain"]
    # Count the votes excluding abstentions.
    unit, counts = np.unique(votes_valid, return_counts=True)
    return unit[np.argmax(counts)]


def extract_name_and_id(name: str) -> tuple[str, int]:
    """extract player name and id from a string"""
    try:
        name = re.search(r"\b[Pp]layer\d+\b", name).group(0)
        idx = int(re.search(r"[Pp]layer(\d+)", name).group(1)) - 1
    except AttributeError:
        # In case Player remains silent or speaks to abstain.
        logger.warning(f"vote: invalid name {name}, set to Abstain")
        name = "Abstain"
        idx = -1
    return name, idx


def extract_scientist_names(name: str) -> list:
    """extract player name and id from a string"""
    try:
        matches = re.findall(r"\b[Ss]cientist\d+\b", name)
        # idx = int(re.search(r"[Pp]layer(\d+)", name).group(1)) - 1
        names = [f"{num}" for num in matches]
    except AttributeError:
        # In case Player remains silent or speaks to abstain.
        logger.warning(f"vote: invalid name {name}, set to Abstain")
        names = ["Abstain"]
        idx = -1
    return list(set(names))


def team_description(team: list, over_state: int) -> str:
    """combine agent names into a string, and use "and" to connect the last
    two names."""
    output_string = "{"
    i = 1
    for team_index in team:
        if team_index.state != over_state:
            output_string += f"team{i}: {team_index.teammate}"
            i = i + 1
            if i < len(team):
                output_string += ", "
    output_string += "}"

    return output_string

def team_description_detail(team: list, agent_list: list, over_state: int) -> str:
    """combine agent names into a string, and use "and" to connect the last
    two names."""
    output_string = ""
    i=1
    for team_index in range(len(team)):
        if team[team_index].state!=over_state:
            team_list = team[team_index].teammate
            output_string += f"The Team{i} includes team members: {team_list}. "
            i=i+1
    output_string_before = f"You are currently a member of {i-1} teams. "
    # for agent in agent_list:
    #     if agent.name in independent_list:
    #         output_string += f"For {agent.name}, "
    #         output_string += convert_you_to_other(agent.sys_prompt)
    output_string = output_string_before + output_string + "Summarize the status of all the teams you are currently part of."
    return output_string

def convert_you_to_other(origin: str) -> str:
    after = origin.replace("Your", "His")
    after = after.replace("you", "he")
    after = after.replace("your", "his")
    return after

# def paper_search(query : str,
#                  top_k : int = 8,
#                  start_year : int = None,
#                  end_year : int = None,
#                  search_engine : str = 'arxiv') -> list:
#     """Given a query, retrieve k abstracts of similar papers from google scholar"""

#     proxy = {
#         'http':'http://u-cEoRwn:EDvFuZTe@172.16.4.9:3128',
#         'https':'http://u-cEoRwn:EDvFuZTe@172.16.4.9:3128',
#     }


#     start_year = 0 if start_year is None else start_year
#     end_year = 9999 if end_year is None else end_year
#     papers = []
#     if search_engine == 'google scholar':
#         # retrieval_results = scholarly.search_pubs(query)
#         retrieval_results = []
#     elif search_engine == 'dblp':
#         retrieval_results = dblp_search_publications(query, num_results = top_k)['content']
#     else:
#         temp_results = arxiv_search(query, max_results = top_k, proxy = proxy).content
#         if isinstance(temp_results, dict):
#             retrieval_results = temp_results['entries']
#         else:
#             retrieval_results = []
#             print(temp_results)

#     for paper in retrieval_results:

#         if len(papers) >= top_k:
#             break

#         try:
#             pub_year = paper.get('published', None)[:4]
#         except:
#             pub_year = paper.get('year', None)

#         if pub_year and start_year <= int(pub_year) <= end_year:
#             if search_engine == 'google scholar':
#                 paper_info = {
#                     'title': paper.get('title'),
#                     'authors': paper.get('authors'),
#                     'year': pub_year,
#                     'abstract': paper.get('abstract'),
#                     'url': paper.get('url'),
#                     'venue': paper.get('venue')
#                 }
#             elif search_engine == 'dblp':
#                 paper_info = {
#                     'title': paper.get('title'),
#                     'authors': paper.get('authors'),
#                     'year': pub_year,
#                     'abstract': paper.get('abstract'),
#                     'url': paper.get('url'),
#                     'venue': paper.get('venue')
#                 }
#             else:
#                 paper_info = {
#                     'title': paper.get('title'),
#                     'authors': ','.join(paper.get('authors')),
#                     'year': pub_year,
#                     'abstract': paper.get('summary'),
#                     'pdf_url': paper.get('url'),
#                     'venue': paper.get('comment')
#                 }

#             # print(paper_info)
#             papers.append(paper_info)

#     return papers

# def paper_search(query : str,
#                  top_k : int = 8,
#                  start_year : int = None,
#                  end_year : int = None,
#                  search_engine : str = 'arxiv') -> list:
#     """Given a query, retrieve k abstracts of similar papers from google scholar"""
#
#     proxy = {
#         'http':'http://u-cEoRwn:EDvFuZTe@172.16.4.9:3128',
#         'https':'http://u-cEoRwn:EDvFuZTe@172.16.4.9:3128',
#     }
#
#     start_year = 0 if start_year is None else start_year
#     end_year = 9999 if end_year is None else end_year
#     papers = []
#     if search_engine == 'google scholar':
#         # retrieval_results = scholarly.search_pubs(query)
#         retrieval_results = []
#     elif search_engine == 'dblp':
#         retrieval_results = dblp_search_publications(query, num_results = top_k)['content']
#     else:
#         temp_results = arxiv_search(query, max_results = top_k, proxy = proxy).content
#         if isinstance(temp_results, dict):
#             retrieval_results = temp_results['entries']
#         else:
#             retrieval_results = []
#             print(temp_results)
#
#     for paper in retrieval_results:
#
#         if len(papers) >= top_k:
#             break
#
#         try:
#             pub_year = paper.get('published', None)[:4]
#         except:
#             pub_year = paper.get('year', None)
#
#         if pub_year and start_year <= int(pub_year) <= end_year:
#             if search_engine == 'google scholar':
#                 paper_info = {
#                     'title': paper.get('title'),
#                     'authors': paper.get('authors'),
#                     'year': pub_year,
#                     'abstract': paper.get('abstract'),
#                     'url': paper.get('url'),
#                     'venue': paper.get('venue')
#                 }
#             elif search_engine == 'dblp':
#                 paper_info = {
#                     'title': paper.get('title'),
#                     'authors': paper.get('authors'),
#                     'year': pub_year,
#                     'abstract': paper.get('abstract'),
#                     'url': paper.get('url'),
#                     'venue': paper.get('venue')
#                 }
#             else:
#                 url = paper.get('entry_id')
#                 # Regular expression to extract arXiv ID
#                 pattern = r'arxiv\.org\/abs\/([0-9]+\.[0-9]+)'
#
#                 # Find the arXiv ID
#                 match = re.search(pattern, url)
#                 arxiv_id = match.group(1)
#
#                 paper_info = {
#                     'title': paper.get('title'),
#                     'authors': ','.join(paper.get('authors')),
#                     'year': pub_year,
#                     'abstract': paper.get('summary'),
#                     'pdf_url': paper.get('url'),
#                     'venue': paper.get('comment'),
#                     'arxiv_id': arxiv_id
#                 }
#
#             # print(paper_info)
#             papers.append(paper_info)
#
#     return papers


# def process_file(file_path, id):
#     try:
#         print(id)
#         with open(file_path, 'r') as file:
#             file_content = file.read()
#             file_dict_old = eval(file_content)
#             file_dict = {
#                 'title': file_dict_old.get('title'),
#                 'abstract': file_dict_old.get('abstract'),
#                 'id': id,
#                 'authors': None,
#                 'cite_papers': None
#             }
#             return file_dict
#     except json.JSONDecodeError:
#         print(f"文件 {file_path} 的内容不是有效的JSON格式，跳过该文件。")
#         return None

# def read_txt_files_as_dict(folder_path):
#     dict_list = []
#     id = 0
#     with ThreadPoolExecutor() as executor:
#         # 只处理 .txt 文件
#         txt_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(".txt")]

#         # 并行处理文件
#         results = list(tqdm(executor.map(lambda file: process_file(file, id), txt_files), total=len(txt_files)))

#         # 过滤掉 None 结果
#         dict_list = [res for res in results if res is not None]

#     return dict_list

def read_txt_files_as_dict(folder_path):
    dict_list = []  # 用于存储所有文件的字典
    id = 0
    # 遍历文件夹中所有的 .txt 文件
    for filename in tqdm(os.listdir(folder_path)):
        if filename.endswith(".txt"):  # 确保文件是 .txt 类型
            file_path = os.path.join(folder_path, filename)

            # 打开并读取每个 .txt 文件的内容
            with open(file_path, 'r') as file:
                file_content = file.read()
                try:
                    # 将内容解析为字典（假设内容是JSON格式）
                    file_dict_old = eval(file_content)
                    file_dict={}
                    file_dict['title']=file_dict_old['title']
                    file_dict['abstract']=file_dict_old['abstract']
                    file_dict['year']=file_dict_old['year']
                    file_dict['citation']=file_dict_old['citation']
                    file_dict['id'] = id
                    file_dict['authors'] = None
                    file_dict['cite_papers'] = None
                except json.JSONDecodeError:
                    print(f"文件 {filename} 的内容不是有效的JSON格式，跳过该文件。")
                    continue

                # 将字典添加到列表
                dict_list.append(file_dict)
                id = id + 1
    return dict_list

def extract_between_json_tags(text, num=None):
    json_blocks = re.findall(r'```json(.*?)```', text, re.DOTALL)

    if not json_blocks:
        start_tag = '```json'
        end_tag = '```'
        start_idx = text.find(start_tag)
        end_idx = text.find(end_tag, start_idx + len(start_tag))
        return text[start_idx + len(start_tag):].strip()
    else:
        if num==None:
            combined_json = ''.join(block.strip() for block in json_blocks)
        else:
            combined_json = ''.join(block.strip() for block in json_blocks[:num])
        return combined_json


def extract_metrics(text, split_keywords):
    # 存储每个指标及其数值
    metrics = {}

    for keyword in split_keywords:
        # 使用关键词进行分割
        parts = text.split(keyword)
        if len(parts) > 1:
            # 在分割后的部分中找到第一个数字
            match = re.search(r'\d+', parts[1])
            if match:
                value = int(match.group())
                metrics[keyword.strip('"')] = value
            else:
                metrics[keyword.strip('"')] = None
        else:
            metrics[keyword.strip('"')] = None

    return metrics

def strip_non_letters(text):
    # 正则表达式匹配非字母字符并移除它们
    return re.sub(r'^[^a-zA-Z]+|[^a-zA-Z]+$', '', text)

def save2database(paper_list : list[dict], output_dir : str):
    # connect to database, if it exists then delete it
    if os.path.isfile(output_dir):
        os.remove(output_dir)
    conn = sqlite3.connect(output_dir)
    # create cursor
    cursor = conn.cursor()

    # create table
    print('build paper table...')
    cursor.execute('''
            CREATE TABLE IF NOT EXISTS papers (
                id INTEGER PRIMARY KEY,
                title TEXT,
                authors TEXT,
                cite_papers TEXT,
                abstract TEXT
            )
        ''')

    for paper in paper_list:
        id = int(paper['id'])
        title = paper['title']
        abstract = paper['abstract']
        if paper['authors']!=None:
            authors = ';'.join(paper['authors'])
            paper_references = ';'.join(map(str, paper['cite_papers']))
        else:
            authors = None
            paper_references = None

        # Define your user data (id, name, affiliation)
        paper_data = (id,
                      title,
                      authors,
                      paper_references,
                      abstract)

        # Insert query
        query = '''
            INSERT INTO papers (id, title, authors, cite_papers, abstract)
            VALUES (?, ?, ?, ?, ?)
            '''

        # Execute the query with user data
        cursor.execute(query, paper_data)

    conn.commit()
    cursor.close()
    conn.close()

def count_team(team_list: list[dict], over_state: int):
    num = 0
    for team in team_list:
        if team.state<over_state:
            num = num+1
    return num


def top_three_indices(lst):
    # 使用enumerate获取元素及其索引，并根据元素值进行排序
    sorted_indices = sorted(enumerate(lst), key=lambda x: x[1], reverse=True)

    # 取出前三大的元素的索引
    top_three = [index for index, value in sorted_indices[:3]]

    return top_three

def extract_first_number(s):
    # 使用正则表达式查找字符串中的第一个数字
    match = re.search(r'\d+', s)
    if match:
        return match.group()  # 返回匹配到的第一个数字
    return None  # 如果没有找到数字，返回 None


def most_frequent_element(arr):
    # 使用 Counter 计算每个元素的出现次数
    count = Counter(arr)

    # 返回出现次数最多的元素
    most_common_element = count.most_common(1)[0][0]

    return most_common_element

def read_txt_files_as_list(folder_path):
    dict_list = []  

    for filename in tqdm(os.listdir(folder_path)):
        if filename.endswith(".txt"):  
            file_path = os.path.join(folder_path, filename)
            with open(file_path, 'r') as file:
                file_content = file.read()
                dict_list.append(file_content)
    return dict_list

def process_author_text(input_text):
    # 拆分出各个部分信息
    name_start = input_text.find("Your name is")
    name_end = input_text.find(",", name_start)
    name = input_text[name_start + len("Your name is "):name_end]

    # 处理affiliations部分
    affiliations_start = input_text.find("following affiliations")
    affiliations_end = input_text.find("],", affiliations_start)
    affiliations = input_text[affiliations_start + len("following affiliations "):affiliations_end + 1]
    affiliations = affiliations.replace("['", "").replace("']", "").replace("', '", "; ")

    # 处理research topics部分
    topics_start = input_text.find("following topics")
    topics_end = input_text.find("],", topics_start)
    topics = input_text[topics_start + len("following topics "):topics_end + 1]
    topics = topics.replace("['", "").replace("']", "").replace("', '", ", ")

    # 处理发表论文数和引用数
    papers_start = input_text.find("published")
    papers_end = input_text.find("papers", papers_start)
    num_papers = input_text[papers_start + len("published "):papers_end].strip()

    citations_start = input_text.find("have", papers_end)
    citations_end = input_text.find("citations", citations_start)
    num_citations = input_text[citations_start + len("have "):citations_end].strip()

    # 处理合作伙伴部分
    collaborators_start = input_text.find("these individuals")
    collaborators_end = input_text.find("].", collaborators_start)
    collaborators = input_text[collaborators_start + len("these individuals "):collaborators_end + 1]
    collaborators = collaborators.replace("['", "").replace("']", "").replace("', '", ", ")

    # 拼接客观表述的结果
    output_text = (
        f"{name} is affiliated with the following institutions: {affiliations}. "
        f"Their research has focused on topics such as {topics}. "
        f"{name} has published {num_papers} papers and has received {num_citations} citations. "
        f"Previous collaborations include work with individuals such as {collaborators}."
    )
    return output_text