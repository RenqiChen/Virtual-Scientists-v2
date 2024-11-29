import json
import os
import ast
import ollama
import re
import difflib
import csv
import json
from tqdm import tqdm

def find_best_match(target, options, cutoff = 0.0):
    # Find the best match
    best_match = difflib.get_close_matches(target, options, n=1, cutoff=cutoff)
    return best_match[0] if best_match else None

def filter_out_number_n_symbol(text):
    filtered_text = re.sub(r'[^\w\s,&]', '', text)  # Remove symbols
    filtered_text = ''.join([char for char in filtered_text if not char.isdigit()])  # Remove numbers
    filtered_text = filtered_text.strip()  # Remove leading/trailing whitespace
    return filtered_text

def predict_field(discipline: str):
    discipline2field = {
        'art': 'Humanities, Literature & Arts',
        'biology': 'Life Sciences & Earth Sciences',
        'business': 'Business, Economics & Management',
        'computer science': 'Engineering & Computer Science',
        'chemistry': 'Chemical & Material Sciences',
        'economics': 'Business, Economics & Management',
        'engineering': 'Engineering & Computer Science',
        'environmental science': 'Life Sciences & Earth Sciences',
        'geography': 'Life Sciences & Earth Sciences',
        'geology': 'Life Sciences & Earth Sciences',
        'history': 'Humanities, Literature & Arts',
        'materials science': 'Chemical & Material Sciences',
        'mathematics': 'Physics & Mathematics',
        'medicine': 'Health & Medical Sciences',
        'philosophy': 'Humanities, Literature & Arts',
        'physics': 'Physics & Mathematics',
        'political science': 'Social Sciences',
        'psychology': 'Humanities, Literature & Arts',
        'sociology': 'Social Sciences',
    }

    return discipline2field[discipline.lower()]

def predict_discipline(key_words: list, ollama_client):
    disciplines_prompt = '''
    Given the following key words of a paper, identify the most likely one discipline from the list below:
    
    Disciplines:
    1. Art
    2. Biology
    3. Business
    4. Computer Science
    5. Chemistry
    6. Economics
    7. Engineering
    8. Environmental science
    9. Geography
    10. Geology
    11. History
    12. Materials Science
    13. Mathematics
    14. Medicine
    15. Philosophy
    16. Physics
    17. Political Science
    18. Psychology
    19. Sociology
    
    Paper's Key Words:
    KEY_WORDS
    
    Reply Format:
    [Selected Discipline]
    
    Only one discipline should be selected! Do not reply anything else!!!!!'''

    key_words = ', '.join(key_words)
    prompt = disciplines_prompt.replace('KEY_WORDS', key_words)

    # predicted discipline by LLM
    discipline = ollama_client.chat(model='llama3.1', messages=[
        {
            'role': 'user',
            'content': prompt,
        }
    ], options={'temperature': 0})

    disciplines = ['art', 'biology', 'business', 'computer science', 'chemistry', 'economics', 'engineering', 'environmental science',
                   'geography', 'geology', 'history', 'materials science', 'mathematics', 'medicine', 'philosophy', 'physics', 'political science',
                   'psychology', 'sociology']

    discipline = find_best_match(filter_out_number_n_symbol(discipline['message']['content'].lower()), disciplines)

    return discipline

def predict_discipline_abstract(abstract: str, ollama_client):
    disciplines_prompt = '''
    Given the following abstract of a paper, identify the most likely one discipline from the list below:
    
    Disciplines:
    1. Art
    2. Biology
    3. Business
    4. Computer Science
    5. Chemistry
    6. Economics
    7. Engineering
    8. Environmental science
    9. Geography
    10. Geology
    11. History
    12. Materials Science
    13. Mathematics
    14. Medicine
    15. Philosophy
    16. Physics
    17. Political Science
    18. Psychology
    19. Sociology
    
    Paper's Abstract:
    ABSTRACT
    
    Reply Format:
    [Selected Discipline]
    
    Only one discipline should be selected! Do not reply anything else!!!!!'''

    prompt = disciplines_prompt.replace('ABSTRACT', abstract)

    # predicted discipline by LLM
    discipline = ollama_client.chat(model='llama3.1', messages=[
        {
            'role': 'user',
            'content': prompt,
        }
    ], options={'temperature': 0})

    disciplines = ['art', 'biology', 'business', 'computer science', 'chemistry', 'economics', 'engineering', 'environmental science',
                   'geography', 'geology', 'history', 'materials science', 'mathematics', 'medicine', 'philosophy', 'physics', 'political science',
                   'psychology', 'sociology']
    
    discipline = find_best_match(filter_out_number_n_symbol(discipline['message']['content'].lower()), disciplines)

    return discipline

def assign_ranking_by_year(
        file_path = "/home/bingxing2/ailab/group/ai4agr/crq/SciSci/OAG/v3.1_oag_affiliation.json",
        ranking_file_path = "THE World University Rankings 2016-2025.csv",
        year = 2025
):
    # Load ranking file as a dictionary
    affiliation_ranking_dict = {}
    with open(ranking_file_path, mode="r", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            if row["Year"] == f"{year}":  # Filter by Year
                affiliation_ranking_dict[row["Name"]] = int(float(row["Rank"]))

    # load all affiliations
    with open(file_path, 'r', encoding='utf-8') as f:
        total_lines = sum(1 for _ in f)

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f, total=total_lines, desc="Processing lines"):
            affiliation = ast.literal_eval(line.strip())
            match = find_best_match(affiliation['name'], affiliation_ranking_dict.keys(), 0.6)
            if match is not None:
                affiliation['ranking'] = affiliation_ranking_dict[match]
            else:
                affiliation['ranking'] = -1.0

            # output new affiliation info
            with open(f"{year}_new_v3.1_oag_affiliation.txt", "a") as file:
                file.write(json.dumps(affiliation) + "\n")

async def assign_discipline_and_field(root_dir, file_name, ollama_client, year_range = None):
    # 逐行读取大文件
    with open(f'{root_dir}/{file_name}', 'r', encoding='utf-8') as f:
        total_lines = sum(1 for _ in f)
    with open(f'{root_dir}/{file_name}', 'r', encoding='utf-8') as f:
        for line in tqdm(f, total=total_lines, desc="Processing lines"):
            paper_info = ast.literal_eval(line.strip())
            if year_range is not None and (paper_info['year'] < year_range[0] or paper_info['year'] > year_range[1]):
                continue
            else:
                discipline = predict_discipline(paper_info['keywords'], ollama_client)
                field = predict_field(discipline)
                paper_info['discipline'] = discipline
                paper_info['field'] = field
                with open(f"{root_dir}/new_{file_name}", "a") as file:
                    file.write(json.dumps(paper_info) + "\n")

if __name__ == '__main__':
    key_words = ['machine learning']
    abstract = '''
    Title: Revolutionizing Caries Management in Primary Molars using Advanced Imaging, AI-Powered Decision Support, and Minimally Invasive Treatments
    Abstract: Dental caries remains a pervasive public health concern worldwide, affecting millions of children annually. The inadequacy of traditional restorative treatments has led to persistent pain, infection, and costly follow-up appointments in pediatric patients, particularly those with primary molars. 
    Our innovative protocol integrates cutting-edge technologies (cone beam computed tomography, intraoral cameras) with machine learning algorithms for personalized treatment planning, and minimally invasive treatments to minimize discomfort and promote healing. This patient-centered approach aims to provide more effective and efficient care for pediatric patients worldwide.
    To evaluate the efficacy of our protocol, we conducted rigorous randomized controlled trials in a diverse cohort of children (n=500). Our results demonstrate that this novel protocol significantly reduces failure rates (by 42%), pain (by 32%), and the number of dental visits required for follow-up appointments (by 25%). Patient satisfaction is also improved, as measured by standardized questionnaires.
    Moreover, our research highlights the critical need for continued investment in dental research and innovation. By harnessing the collective expertise of dentists, researchers, policymakers, and industry partners, we can accelerate progress towards achieving optimal oral health outcomes for all children worldwide.
    Our protocol prioritizes pain management, anxiety reduction, and educational empowerment to promote healthy oral habits and prevent future caries. This holistic approach is grounded in the principles of shared decision-making and personalized medicine, ensuring that each child receives tailored care that respects their unique needs and circumstances.
    The scalability and adaptability of our protocol are critical factors in its potential impact on global public health. By integrating advanced imaging techniques, AI-powered decision support, and minimally invasive treatments into standard care protocols, we can ensure that all children have access to high-quality dental care, regardless of geographical or socio-economic constraints.
    In conclusion, our research presents a paradigm shift in caries management for primary molars, offering a more effective, efficient, and patient-centered approach. By harnessing the power of advanced technologies and evidence-based practices, we can revolutionize the way we address this critical public health concern and promote optimal oral health outcomes for all children worldwide.
    '''
    ollama_client = ollama.Client(host='127.0.0.1:11434')
    print(predict_discipline_abstract(abstract, ollama_client))
    # print(predict_discipline(key_words, ollama_client))
    # with open('/home/bingxing2/ailab/suhaoyang/shy/Social_Science_CAMEL/preprocess_data/OAG_data/new_v3.1_oag_publication_1.txt', 'r', encoding='utf-8') as f:
    #     total_lines = sum(1 for _ in f)
    # print(total_lines)
    # assign_ranking_by_year()
