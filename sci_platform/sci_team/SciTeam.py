from datetime import datetime

import logging
import re
import ollama
import torch.nn.functional
import numpy as np
import json
import os
import sys
sys.path.append('../camel-master')

from camel.messages import BaseMessage

from utils.prompt import Prompts
from utils.scientist_utils import (
    extract_scientist_names,
    team_description,
    convert_you_to_other,
    team_description_detail,
    read_txt_files_as_dict,
    extract_between_json_tags,
    extract_metrics,
    strip_non_letters,
    save2database,
    count_team,
    top_three_indices,
    extract_first_number,
    most_frequent_element,
    Color
)

class Team:
    def __init__(self, team_name, log_dir, info_dir, recent_n_team_mem_for_retrieve):
        # attrs
        self.team_name = team_name
        self.state = 1
        self.epoch = -1
        self.teammate = []
        self.memory = []
        self.recent_n_team_mem_for_retrieve = recent_n_team_mem_for_retrieve
        self.topic = None
        self.idea = None
        self.abstract = None
        self.citation_id = None
        self.self_review = None
        self.paper_review = None

        # state log
        self.state_log = {
            1: 'WAIT',
            2: 'TOPIC',
            3: 'IDEA',
            4: 'CHECK',
            5: 'ABSTRACT',
            6: 'REVIEW',
            7: 'FINISH'
        }

        # state action
        self.state_action = {
            2: self.select_topic,
            3: self.generate_idea,
            4: self.check_novelty,
            5: self.generate_abstract,
            6: self.generate_review
        }

        # init log file dir
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.info_file = f"{info_dir}/{current_time}_{self.team_name}_dialogue.json"
        self.log_file = f"{log_dir}/{current_time}_{self.team_name}_dialogue.log"

        # Check if log file exists and delete it
        if os.path.exists(self.log_file):
            os.remove(self.log_file)

        self.logger = logging.getLogger(self.team_name)
        self.logger.setLevel(logging.INFO)
        fh = logging.FileHandler(self.log_file)
        self.logger.addHandler(fh)

    # format memories
    def format_memories(self, current_memories = None, previous_memories = None, team_memories = None):
        memory_type_hint = ['Discussion in this turn', 'Summarization of previous turns', 'Team memory']

        output = ''

        if current_memories is not None and len(current_memories) != 0:
            format_current_memories = f'{memory_type_hint[0]}:\n'
            for memory in current_memories:
                format_current_memories += (memory.role_name + ': ' + memory.content + '\n')

            output = format_current_memories + output

        if previous_memories is not None and len(previous_memories) != 0:
            format_previous_memories = f'{memory_type_hint[1]}:\n'
            for memory in previous_memories:
                format_previous_memories += (memory.role_name + ': ' + memory.content + '\n')

            output = format_previous_memories + output

        if team_memories is not None and len(team_memories) != 0:
            format_team_memories = f'{memory_type_hint[2]}:\n'
            for memory in team_memories[-self.recent_n_team_mem_for_retrieve:]:
                format_team_memories += (memory.content + '\n')

            output = format_team_memories + output

        return output


    # execute action based on team state
    def action_excution(self, platform):
        if self.state in self.state_log.keys():
            print(f'{"="*50} Epoch: {self.epoch} | BEGIN {self.state_log[self.state]} PROCESS {"="*50}')

        action = self.state_action.get(self.state, None)
        if action is not None:
            action(platform)
            # print(f'{"="*50} Epoch: {self.epoch} | FINISH {self.state_log[self.state]} PROCESS {"="*50}')

    # general group discussion process
    def group_discuss(self, platform, prompt: str = None):
        # prompt is used to start and guide the discussion
        # for each turn, in group_discuss, all dialogue history is stored in dialogue_history but not in agent memory
        # after finishing each discussion turn, agent1 will summarize dialogue_history and add a summarization into team_history

        # get teammate
        teammate = platform.id_to_agent(self.teammate)

        # Memoryies
        # get team_history, previous group discussion summarizations
        team_memories = self.memory
        # init previous_memories, a list of summarizations of previous turns in this group discussion
        previous_memories = []

        # init exit state
        exit = False
        # output return dialogue history, summarization of the last turn, and memory of the last turn
        output = {}
        # start discussing
        if len(teammate) == 1:
            group_max_discuss_iteration = platform.group_max_discuss_iteration
        else:
            group_max_discuss_iteration = platform.group_max_discuss_iteration

        for turn in range(group_max_discuss_iteration):
            # init current_memories for each turn
            current_memories = []

            said = []
            agent_num = 0
            for agent in teammate:
                if agent.role_name in said:
                    continue
                else:
                    said.append(agent.role_name)

                agent_prompt = f"Current team members are {','.join(self.teammate)}.\n" + \
                               self.format_memories(None, None, team_memories) + \
                               prompt + \
                               self.format_memories(current_memories, previous_memories, None)
                format_agent_prompt = BaseMessage.make_user_message(role_name="user", content=agent_prompt)

                # add reply to turn_history
                reply = agent.step(format_agent_prompt).msg
                if reply.content != None and len(reply.content) > 0:
                    self.log_dialogue(agent.role_name, reply.content)
                involved_scientist = extract_scientist_names(reply.content)
                print(involved_scientist)

                # judge whether someone is called to join the team
                for scientist_index in involved_scientist:
                    if scientist_index not in self.teammate:
                        if "by the way" in reply.content or "By the way" in reply.content:
                            # hint = BaseMessage.make_user_message(role_name="user", content=agent_prompt)
                            special_guest_prompt = agent_prompt + \
                                                reply.role_name + ': ' + reply.content + '\n'
                            format_special_guest_prompt = BaseMessage.make_user_message(role_name="user", content=special_guest_prompt)
                            # invite new team member to comment
                            special_guest_reply = platform.id2agent[scientist_index].step(format_special_guest_prompt).msg
                            if special_guest_reply.content is not None:
                                said.append(scientist_index)
                                self.teammate.append(scientist_index)
                                self.log_dialogue(platform.id2agent[scientist_index].role_name, special_guest_reply.content)
                                teammate.append(platform.id2agent[scientist_index])

                current_memories.append(reply)
                agent_num = agent_num + 1
                # discussion is finished
                if 'exit' in reply:
                    exit = True
                    break

            # summarize this turn's discussion
            turn_summarization_prompt = 'Briefly summarize "Discussion in this turn".' +\
                                        self.format_memories(current_memories, previous_memories, None)
            format_turn_summarization_prompt = BaseMessage.make_user_message(role_name="user", content=turn_summarization_prompt)

            x = teammate[0].step(format_turn_summarization_prompt).msg
            self.log_dialogue(teammate[0].role_name, x.content)
            turn_summarization = BaseMessage.make_user_message(role_name="Summarization of turn{}".format(turn+1), content=x.content)

            if exit or turn==group_max_discuss_iteration-1:
                output['last_turn_summarization'] = turn_summarization
                output['last_turn_history'] = current_memories
                break
            else:
                print(turn_summarization)
                previous_memories.append(turn_summarization)

        output['previous_memories'] = previous_memories
        self.teammate = platform.agent_to_id(teammate)
        return output

    def select_topic(self, platform):
        # prompt to start discussing select_topic
        discuss_result = self.group_discuss(platform, Prompts.to_start_topic_discussion)
        team_memories = self.memory
        previous_memories = discuss_result['previous_memories']
        current_memories = discuss_result['last_turn_history']
        last_turn_summarization = discuss_result['last_turn_summarization']

        answer_prompt = self.format_memories(current_memories, previous_memories, team_memories) +\
                        Prompts.to_ask_if_ready_give_topic
        format_answer_prompt = BaseMessage.make_user_message(role_name="user", content=answer_prompt)

        answer = platform.id2agent[self.teammate[0]].step(format_answer_prompt).msg
        # self.log_dialogue('user', answer_prompt)
        self.log_dialogue(platform.id2agent[self.teammate[0]].role_name, answer.content)
        answer_pattern = re.compile(r'action\s*1', re.IGNORECASE)

        # check whether agent is ready to answer
        if answer_pattern.search(answer.content) or len(team_memories)>=1:
            self.state = 3
            # prompt
            topic_prompt = Prompts.to_ask_topic.replace("[history_prompt]", self.format_memories(current_memories, previous_memories, team_memories))
            format_topic_prompt = BaseMessage.make_user_message(role_name="user", content=topic_prompt)
            # answer
            topic = platform.id2agent[self.teammate[0]].step(format_topic_prompt).msg
            self.log_dialogue(self.teammate[0], topic.content)
            self.topic = extract_between_json_tags(topic.content, num=1)
            self.topic = strip_non_letters(self.topic.split("Topic")[1])
            # update dialogue history
            previous_memories.append(last_turn_summarization)
            topic_message = BaseMessage.make_user_message(role_name="user", content="Final selected topic: "+self.topic)
            previous_memories.append(topic_message)
        else:
            # update dialogue history
            previous_memories.append(last_turn_summarization)
        # summarize dialogue history
        dialogue_summarization_prompt = 'Briefly summarize "Summarizations of previous turns".' + \
                                        self.format_memories(None, previous_memories, team_memories)
        format_dialogue_summarization_prompt = BaseMessage.make_user_message(role_name="user", content=dialogue_summarization_prompt)
        dialogue_summarization = platform.id2agent[self.teammate[0]].step(format_dialogue_summarization_prompt).msg
        team_memories.append(dialogue_summarization)
        self.memory = team_memories

    def generate_idea(self, platform):
        topic = self.topic
        old_idea = None
        best_idea = None
        idea_list = []
        mark_list = []
        # search related paper about the topic
        selected_topics = strip_non_letters(topic.split("Selected Topics:")[-1])
        paper_reference, cite_paper = platform.reference_paper(selected_topics, platform.cite_number)

        teammate = platform.id_to_agent(self.teammate)
        idea_judge = True

        if len(teammate)==1:
            group_max_discuss_iteration = platform.group_max_discuss_iteration
        else:
            group_max_discuss_iteration = platform.group_max_discuss_iteration

        for turn in range(group_max_discuss_iteration):
            # discuss the idea
            for agent in teammate:
                idea_prompt = Prompts.prompt_task+Prompts.prompt_existing_idea.format(old_idea) + \
                              Prompts.prompt_topic.format(selected_topics)+Prompts.prompt_reference.format(paper_reference) + \
                              Prompts.prompt_response

                format_idea_prompt = BaseMessage.make_user_message(role_name="user", content=idea_prompt)
                reply = agent.step(format_idea_prompt).msg
                # self.log_dialogue('user', idea_prompt)
                self.log_dialogue(agent.role_name, reply.content)
                old_idea = extract_between_json_tags(reply.content, num=1)
                if "Title" in old_idea:
                    idea_key = old_idea.split("Title")[1]
                    idea_key = strip_non_letters(idea_key.split("Experiment")[0])
                else:
                    idea_key = old_idea.split("Idea")[1]
                    idea_key = strip_non_letters(idea_key.split("Experiment")[0])
                paper_reference, cite_paper_new = platform.reference_paper(idea_key, platform.cite_number)
                cite_paper = list(set(cite_paper).union(cite_paper_new))

                # find the metric
                split_keywords = ['Clarity', 'Feasibility', 'Novelty']
                metrics = extract_metrics(old_idea, split_keywords)
                if best_idea != None:
                    if old_idea == best_idea:
                        idea_judge=True
                        print("exit early!!!!!!")
                        break
                    best_metrics = extract_metrics(best_idea, split_keywords)
                    old_count = 0
                    best_count = 0
                    for split_keywork in split_keywords:
                        if metrics[split_keyword]==None:
                            break
                        if split_keyword=='Novelty':
                            old_count = old_count + 2*metrics[split_keyword]
                        else:
                            old_count = old_count + metrics[split_keyword]
                        if best_metrics[split_keyword]==None:
                            break
                        best_count = best_count + best_metrics[split_keyword]
                    if old_count>=best_count:
                        best_idea = old_idea
                        idea_list.append(old_idea)
                        mark_list.append(old_count)
                else:
                    idea_list.append(old_idea)
                    best_idea = old_idea
                # if all metrics are larger than 8, then over
                for split_keyword in split_keywords:
                    if metrics[split_keyword]==None:
                        break
                    if metrics[split_keyword]<10:
                        idea_judge=False
                        break
                if idea_judge:
                    best_idea=old_idea
                    break
            if idea_judge:
                break
        if self.idea == None:
            if len(idea_list)>3:
                indices = top_three_indices(mark_list)
                idea_list = [idea_list[i] for i in indices]
                self.idea = idea_list
            else:
                self.idea = idea_list
        print("Candidate Idea:")
        print(self.idea)
        if platform.skip_check:
            self.state=5
        else:
            self.state=4
        self.citation_id = cite_paper
        print(len(self.citation_id))

    def check_novelty(self, platform):
        existing_idea = self.idea
        idea_choices = ""
        for idea_index in range(len(existing_idea)):
            idea = existing_idea[idea_index]
            idea_choices = idea_choices+"Idea "+str(idea_index)+":\n"+idea+"\n"
        related_papers = []
        for idea_index in existing_idea:
            title = idea_index.split("Title")[1]
            title = strip_non_letters(title.split("Experiment")[0])
            if len(existing_idea)==3:
                cite_number = 3
            else:
                cite_number = 5
            _, related_paper = platform.reference_paper(title, cite_number)

            related_papers = list(set(related_papers).union(related_paper))

        paper_reference = ""
        for id in range(len(related_papers)):
            paper_index = related_papers[id]
            paper_reference = paper_reference+"Paper {}:".format(id+1)+"\n"
            paper_reference = paper_reference+"Title: "+platform.paper_dicts[paper_index]['title']+"\n"
            paper_reference = paper_reference+"Abstract: "+platform.paper_dicts[paper_index]['abstract']+"}"+"\n"

        teammate = platform.id_to_agent(self.teammate)
        choice_list = []
        if len(teammate)==1:
            group_max_discuss_iteration = platform.group_max_discuss_iteration
        else:
            group_max_discuss_iteration = platform.group_max_discuss_iteration
        for turn in range(group_max_discuss_iteration):
            # discuss the idea
            for agent in teammate:
                idea_novelty_prompt = Prompts.prompt_idea_check + \
                                      Prompts.prompt_idea_check_response.replace("{existing_idea}", idea_choices).replace("{last_query_results}", paper_reference)
                format_idea_novelty_prompt = BaseMessage.make_user_message(role_name="user", content=idea_novelty_prompt)
                reply = agent.step(format_idea_novelty_prompt).msg
                # self.log_dialogue('user', idea_novelty_prompt)
                self.log_dialogue(agent.role_name, reply.content)
                old_idea = extract_between_json_tags(reply.content, num=1)
                idea_choice = extract_first_number(old_idea)
                if idea_choice == None:
                    idea_choice = 0
                choice_list.append(int(idea_choice))

        final_choice = most_frequent_element(choice_list)
        if final_choice<0 or final_choice>=len(existing_idea):
            final_choice = len(existing_idea)-1
        try:
            self.idea = existing_idea[final_choice]
        except:
            self.idea = existing_idea[0]
        print("Final Idea:")
        print(self.idea)
        self.state=5

    def generate_abstract(self, platform):
        idea = self.idea
        old_abstract = self.abstract
        teammate = platform.id_to_agent(self.teammate)

        if len(teammate)==1:
            group_max_discuss_iteration = platform.group_max_discuss_iteration
        else:
            group_max_discuss_iteration = platform.group_max_discuss_iteration

        for turn in range(group_max_discuss_iteration):
            # discuss the abstract
            for agent in teammate:
                if old_abstract == None:
                    abstract_prompt = Prompts.prompt_abstract+"\n"+\
                                      idea+"\n"+\
                                      Prompts.prompt_abstract_requirement+"\n"+\
                                      Prompts.prompt_abstract_response
                else:
                    # the paper is not reviewed by reviewer
                    if self.paper_review == None:
                        # the paper is not reviewer by the team member
                        if self.self_review == None:
                            prompt_abstract_judgement = Prompts.prompt_abstract_judgement.replace("[Insert abstract here]",old_abstract)
                            abstract_prompt = prompt_abstract_judgement+Prompts.prompt_abstract_revise_response
                        else:
                            prompt_abstract_judgement = Prompts.prompt_abstract_judgement_self.replace("[Insert abstract here]",old_abstract)
                            prompt_abstract_judgement = prompt_abstract_judgement.replace("[Insert self_review comments]", self.self_review)
                            abstract_prompt = prompt_abstract_judgement+Prompts.prompt_abstract_revise_response
                    else:
                        prompt_abstract_judgement = Prompts.prompt_abstract_judgement_after_review.replace("[Insert Reviewer comments]",self.paper_review)
                        prompt_abstract_judgement = prompt_abstract_judgement.replace("[Insert abstract here]",old_abstract)
                        abstract_prompt = prompt_abstract_judgement+Prompts.prompt_abstract_revise_response

                format_abstract_prompt = BaseMessage.make_user_message(role_name="user", content=abstract_prompt)
                reply = agent.step(format_abstract_prompt).msg
                self.log_dialogue(agent.role_name, reply.content)
                old_abstract = extract_between_json_tags(reply.content, num=1)
                if old_abstract == None:
                    old_abstract = reply.content

        related_papers = []

        Abstract = strip_non_letters(old_abstract.split("Abstract")[1])
        query_vector = ollama.embeddings(model="mxbai-embed-large", prompt=Abstract)
        query_vector = np.array([query_vector['embedding']])

        # D_future, I_future = platform.gpu_future_index.search(query_vector, int(platform.cite_number/2))
        D, I = platform.gpu_index.search(query_vector, int(platform.cite_number/2))

        # for id in range(len(I_future[0])):
        #     paper_title = platform.paper_future_dicts[I_future[0][id]]['title']
        #     paper_abstract = platform.paper_future_dicts[I_future[0][id]]['abstract']
        #     paper_year = platform.paper_future_dicts[I_future[0][id]]['year']
        #     paper_citation = platform.paper_future_dicts[I_future[0][id]]['citation']
        #     paper_index = {}
        #     paper_index['title'] = paper_title
        #     paper_index['abstract'] = paper_abstract
        #     paper_index['year'] = paper_year
        #     paper_index['citation'] = paper_citation
        #     related_papers.append(paper_index)

        for id in range(len(I[0])):
            paper_title = platform.paper_dicts[I[0][id]]['title']
            paper_abstract = platform.paper_dicts[I[0][id]]['abstract']
            paper_year = platform.paper_dicts[I[0][id]]['year']
            paper_citation = platform.paper_dicts[I[0][id]]['citation']
            paper_index = {}
            paper_index['title'] = paper_title
            paper_index['abstract'] = paper_abstract
            paper_index['year'] = paper_year
            paper_index['citation'] = paper_citation
            related_papers.append(paper_index)

        # eval with embedding similarity
        abs = []
        our_abs = strip_non_letters(old_abstract.split('Abstract')[1])
        abs.append(ollama.embeddings(model="mxbai-embed-large", prompt=our_abs)['embedding'])
        for paper_id in range(len(related_papers)):
            related_astract = related_papers[paper_id]['abstract']
            abs.append(ollama.embeddings(model="mxbai-embed-large", prompt=related_astract)['embedding'])

        sim = []
        for emb_id in range(1, len(abs)):
            sim.append(torch.nn.functional.cosine_similarity(torch.tensor(abs[0]).unsqueeze(0),
                                                             torch.tensor(abs[emb_id]).unsqueeze(0), dim=-1)[0].item())
        self.log_dialogue('embedding similarity', str(sim))

        self.log_dialogue('faiss_distance', str(D))
        # self.log_dialogue('faiss_distance_future', str(D_future))

        # eval with LLM
        print('related papers:')
        print(len(related_papers))
        if len(related_papers)>0:
            self.log_dialogue('arxiv',related_papers)
        # find paper successfully
        if len(related_papers)>0:
            abstract_check_prompt = Prompts.prompt_abstract_check.replace("[Insert your abstract here]", old_abstract)
            cite_abstract = ""
            word = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T']
            split_keywords = []
            for paper_id in range(len(related_papers)):
                cite_abstract = cite_abstract + str(paper_id+1) + ". Abstract {}: ".format(word[paper_id]) + "Title: " + related_papers[paper_id]['title'] + "\n" + "Abstract: " + related_papers[paper_id]['abstract'] + "\n"
                split_keywords.append('Written Abstract vs {}'.format(word[paper_id]))
            abstract_check_prompt = abstract_check_prompt.replace("[Insert ref abstract here]", cite_abstract)
            abstract_check_prompt = abstract_check_prompt + "\n" + Prompts.prompt_response_check

            format_abstract_check_prompt = BaseMessage.make_user_message(role_name="user", content=abstract_check_prompt)
            reply = teammate[0].step(format_abstract_check_prompt).msg
            self.log_dialogue(teammate[0].role_name, reply.content)
            print("abstract_check:")
            print(split_keywords)
            comparison = extract_between_json_tags(reply.content)
            metric = extract_metrics(comparison, split_keywords=split_keywords)
            abstract_use = True
            for split_keyword in split_keywords:
                if metric[split_keyword]>=90:
                    abstract_use = False
                    self.abstract = old_abstract
                    break
            self.abstract = old_abstract
            print('Final Abstract:')
            print(self.abstract)
            # stop early
            # self.state=7

            # do not stop early

            if abstract_use:
                self.state=6
                self.self_review=None
            # if the abstract is too similar one time, go to revise, otherwise back to generate idea
            else:
                if self.self_review!=None:
                    self.state=3
                    self.idea = None
                    self.abstract = None
                    self.citation_id = None
                    self.self_review = None
                    self.paper_review = None
                else:
                    self.self_review = reply.content

        else:
            print('Check Fail!!!!!!')
            if self.abstract == None:
                self.abstract = old_abstract
                print('Final Abstract:')
                print(self.abstract)
                self.state=6

    def generate_review(self, platform):
        # paper reviewer by reviewer
        print('current reviewing paper from {}'.format(self.teammate))
        old_abstract = self.abstract
        review_prompt = Prompts.prompt_review_require_simple.replace("{paper}", old_abstract)
        mark_sum = 0
        self.paper_review == None
        for _ in range(platform.reviewer_num):
            format_review_prompt = BaseMessage.make_user_message(role_name="user", content=review_prompt)
            reply = platform.reviewer_pool[_].step(format_review_prompt).msg
            self.log_dialogue(platform.reviewer_pool[_].role_name, reply.content)
            split_keywords = ['Overall']
            metric = extract_metrics(reply.content, split_keywords)
            if self.paper_review == None:
                self.paper_review = platform.reviewer_pool[_].role_name+":\n"+reply.content
            else:
                self.paper_review = self.paper_review+"\n"+platform.reviewer_pool[_].role_name+":\n"+reply.content
            for split_keyword in split_keywords:
                if metric[split_keyword] == None:
                    mark_sum = mark_sum + platform.default_mark
                else:
                    mark_sum = mark_sum + metric[split_keyword]
        if mark_sum>=(5*platform.reviewer_num):
            print('paper accept!!!!!!')
            self.state=platform.over_state
            title = old_abstract.split("Abstract")[0]
            title = strip_non_letters(title.split("Title")[1])
            abstract = strip_non_letters(old_abstract.split("Abstract")[1])
            file_dict={}
            file_dict['title']=title
            file_dict['abstract']=abstract
            file_dict['year']=self.epoch
            file_dict['citation']=-1
            file_dict['id'] = len(platform.paper_dicts)
            file_dict['authors'] = self.teammate
            file_dict['cite_papers'] = self.citation_id
            platform.paper_dicts.append(file_dict)
            # add embedding into list
            embedding_list = []
            response = ollama.embeddings(model="mxbai-embed-large", prompt=abstract)
            embedding_list.append(response["embedding"])
            response = np.array(embedding_list)
            platform.gpu_index.add(response)
        else:
            self.state = 5

    def log_dialogue(self, name, content):
        color = Color.GREEN
        name_after = f"{color}{name}{Color.RESET}"
        print(f'{name_after}: {content}')
        print(f'-'*30)
        self.logger.info(f'{"="*50} Epoch:{self.epoch} | {self.state_log[self.state]} | {name} {"="*50}\n{content}')
        # self.logger.info(f'{"="*100}')

    def save_team_info(self):
        team_info = {
            'teammate':self.teammate,
            'topic':self.topic,
            'idea':self.idea,
            'abstract':self.abstract
        }
        # print(f'{"="*50} SAVE TEAM INFO {"="*50}')
        with open(self.info_file, 'w') as json_file:
            json.dump(team_info, json_file, indent=4)

if __name__=='__main__':
    team1 = Team('LPL')
    team2 = Team('LCK')
    team1.log_dialogue('sam', 'LPL win!')
    team2.log_dialogue('tom', 'LCK win!')
    team1.log_dialogue('sam', 'LPL win again !')
    team2.log_dialogue('tom', 'LCK win again !')