import sqlite3
import os
import json
import re
from tqdm import tqdm
import numpy as np
def extract_paper_info(paper, year_range, least_citation, least_author = 3):
    # year info
    year = paper['year'] if isinstance(paper['year'], int) else None
    if year is None or year < year_range[0] or year > year_range[1]:
        return None

    # citation info
    n_citation = paper['n_citation'] if isinstance(paper['n_citation'], int) else None
    if n_citation is None or n_citation < least_citation:
        return None

    # author info
    authors = []
    for author in paper['authors']:
        if author['id'].strip()!='':
            authors.append(author['id'].strip())
    if len(authors) < least_author:
        return None

    # id info
    paper_id = paper['id'].strip()
    if paper_id=='':
        return None

    # title info
    title = paper['title'].strip()
    if title=='':
        return None

    # abstract info
    abstract = paper['abstract'].strip()
    if abstract=='':
        return None

    # keywords info
    keywords = paper['keywords']
    if len(keywords)==0:
        return None

    # reference info
    references = paper['references']
    if len(keywords)==0:
        return None

    # venue
    venue = paper['venue_id']
    if venue=='':
        return None

    return (paper_id, title, abstract,
            ';'.join(keywords), year, ';'.join(authors),
            ';'.join(references), n_citation, venue)

def create_contemp_paper_table(conn, cursor, root_dir, year_range, n_citation):
    cursor.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name='contemp_papers';")
    exists = cursor.fetchone() is not None
    if exists:
        cursor.execute('DROP TABLE IF EXISTS contemp_papers')
        print("Contemporary paper table already exists!")
    else:
        # create table
        print('build contemporary papers table...')
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS contemp_papers (
                id TEXT PRIMARY KEY,
                title TEXT,
                abstract TEXT,
                keywords TEXT,
                year INTEGER,
                authors TEXT,
                paper_references TEXT,
                citation INTEGER,
                venue_id TEXT
            )
        ''')

        valid_authors = []
        for i in tqdm(range(1, 15)):
            count = 0
            # 逐行读取大文件
            with open(f'{root_dir}/v3.1_oag_publication_{i}.json', 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        # 解析每一行的 JSON 对象
                        paper = json.loads(line.strip())
                        # 处理数据（例如，打印第一个字典）
                        paper_data = extract_paper_info(paper, year_range, n_citation)
                        if paper_data is not None:
                            # extract author info
                            valid_authors.extend(paper_data[5].split(';'))
                            # Insert query
                            query = '''
                            INSERT INTO papers (id, title, abstract, keywords, year, authors, paper_references, citation, venue_id)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                            '''

                            # Execute the query with user data
                            cursor.execute(query, paper_data)
                            count += 1
                        else:
                            pass
                    except:
                        pass
            valid_authors = list(set(valid_authors))
            print(f'{count} papers are valid')

        conn.commit()

def create_paper_table(conn, cursor, root_dir, data_dir, year_range, n_citation):
    cursor.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name='papers';")
    exists = cursor.fetchone() is not None
    if exists:
        print("Paper table already exists!")
    else:
        # create table
        print('build paper table...')
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS papers (
                id TEXT PRIMARY KEY,
                title TEXT,
                abstract TEXT,
                keywords TEXT,
                year INTEGER,
                authors TEXT,
                paper_references TEXT,
                citation INTEGER,
                venue_id TEXT
            )
        ''')

        valid_authors = []
        for i in tqdm(range(1, 15)):
            count = 0
            # 逐行读取大文件
            with open(f'{root_dir}/v3.1_oag_publication_{i}.json', 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        # 解析每一行的 JSON 对象
                        paper = json.loads(line.strip())
                        # 处理数据（例如，打印第一个字典）
                        paper_data = extract_paper_info(paper, year_range, n_citation)
                        if paper_data is not None:
                            # extract author info
                            valid_authors.extend(paper_data[5].split(';'))
                            # Insert query
                            query = '''
                            INSERT INTO papers (id, title, abstract, keywords, year, authors, paper_references, citation, venue_id)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                            '''

                            # Execute the query with user data
                            cursor.execute(query, paper_data)
                            count += 1
                        else:
                            pass
                    except:
                        pass
            valid_authors = list(set(valid_authors))
            print(f'{count} papers are valid')

        print(f"{len(valid_authors)} valid authors")
        author_in_range = {}
        for id, author in enumerate(valid_authors):
            author_in_range[str(id)] = author

        with open(f'{data_dir}/author_in_range.json', 'w') as json_file:
            json.dump(author_in_range, json_file, indent=4)

        conn.commit()

def create_graph(agentID2authorID, coauthor_list, data_dir):

    authorID2agentID = {}
    for k,v in agentID2authorID.items():
        authorID2agentID[v] = k

    agent_num = len(agentID2authorID)

    adj_matrix = np.zeros((agent_num, agent_num), dtype=np.int32)
    weight_matrix = np.zeros((agent_num, agent_num), dtype=np.int32)

    for agentID, authorID in tqdm(agentID2authorID.items()):
        coauthors = coauthor_list[agentID]
        for coauthor in coauthors:
            try:
                # symmetry adj_matrix
                adj_matrix[agentID][authorID2agentID[coauthor]] = 1
                adj_matrix[authorID2agentID[coauthor]][agentID] = 1

                weight_matrix[agentID][authorID2agentID[coauthor]] += 1
                weight_matrix[authorID2agentID[coauthor]][agentID] += 1
            except:
                pass

        adj_matrix[agentID][agentID] = 0
        weight_matrix[agentID][agentID] = 0

    np.savetxt('{}/adj_matrix.txt'.format(data_dir), adj_matrix, fmt='%d', delimiter=' ')
    np.savetxt('{}/weight_matrix.txt'.format(data_dir), weight_matrix, fmt='%d', delimiter=' ')

def create_adjacency_matrix(data_dir):
    matrix = np.loadtxt('{}/adj_matrix.txt'.format(data_dir), dtype=int)
    row_sum = np.sum(matrix, axis = -1)
    mean_degree = np.mean(row_sum)
    degree_log = 'Mean degree of one-hop adjacency matrix is {}'.format(int(mean_degree))
    print(degree_log)

    degree_int2word = ['one', 'two', 'three', 'four', 'five']
    A = matrix
    adjacency_matries = [A]
    for degree in range(2, 6):
        adjacency_matries.append(np.clip(np.matmul(adjacency_matries[-1], A), a_min=None, a_max=1))
        matrix = np.zeros(A.shape, dtype=int)
        for adjacency_matrix in adjacency_matries:
            matrix += adjacency_matrix
        matrix = np.clip(matrix, a_min=None, a_max=1)
        row_sum = np.sum(matrix, axis = -1)
        mean_degree = np.mean(row_sum)
        np.savetxt('{}/{}-hop_adj_matrix.txt'.format(data_dir, degree_int2word[degree-1]), matrix, fmt='%d', delimiter=' ')
        degree_log = 'Mean degree of {}-hop adjacency matrix is {}'.format(degree_int2word[degree-1], int(mean_degree))
        print(degree_log)

def filter_authors(cursor, data_dir, author_min_degree = 50, author_min_paper = 40):
    cursor.execute(f"SELECT * FROM authors")
    all_authors = cursor.fetchall()

    agentID2authorID = {}
    coauthors = []
    for author in tqdm(all_authors):
        id = author[0]
        paper_num = author[3]
        coauthor = author[-1].split(';')
        degree = len(coauthor)

        if paper_num >= author_min_paper and degree-1 >= author_min_degree:
            agentID2authorID[len(agentID2authorID)] = id
            coauthors.append(coauthor)

    create_graph(agentID2authorID, coauthors, data_dir)
    create_adjacency_matrix(data_dir)

    with open(f'{data_dir}/agentID2authorID.json', 'w') as json_file:
        json.dump(agentID2authorID, json_file, indent=4)


def extract_authors(cursor, root_dir, data_dir):
    with open(f'{data_dir}/author_in_range.json', 'r') as f:
        author_in_range = json.load(f)

    all_authors = {}
    with open(f'{root_dir}/v3.1_oag_author.json', 'r') as f:
        for line in f:
            try:
                data = json.loads(line)
                all_authors[data['id']] = {
                    'name': data['name'],
                    'affiliations': data['org'].strip(),
                    'coauthors': [],
                    'papers': [],
                    'topics': [],
                    'citation_num': 0
                }
            except json.JSONDecodeError as e:
                print(f"Error parsing line: {e}")

    author_candidates = {}
    for k, v in author_in_range.items():
        try:
            author_candidates[v] = all_authors[v]
        except:
            pass
    del all_authors

    # extract publications and coauthors
    cursor.execute(f"SELECT * FROM papers")
    all_papers = cursor.fetchall()

    for paper in tqdm(all_papers):
        id = paper[0]
        coauthors = paper[5].split(';')
        topics = paper[3].split(';')
        citation = int(paper[7])
        for author in coauthors:
            try:
                author_candidates[author]['papers'].append(id)
                author_candidates[author]['coauthors'].extend(coauthors)
                author_candidates[author]['topics'].extend(topics)
                author_candidates[author]['citation_num'] += citation
            except:
                pass

    return author_candidates

def create_author_table(conn, cursor, root_dir, data_dir):
    cursor.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name='authors';")
    exists = cursor.fetchone() is not None
    if exists:
        # cursor.execute('DROP TABLE IF EXISTS authors')
        print("authors table already exists!")
    else:
        # create table
        print('build author table...')
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS authors (
                id TEXT PRIMARY KEY,
                name TEXT,
                affiliations TEXT,
                paper_num INTEGER,
                papers TEXT,
                citation_num INTEGER,
                topics TEXT,
                coauthors TEXT
            )
        ''')

        author_candidates = extract_authors(cursor, root_dir, data_dir)

        # create author table
        for k, v in tqdm(author_candidates.items()):
            v['papers'] = list(set(v['papers']))
            v['coauthors'] = list(set(v['coauthors']))
            v['topics'] = list(set(v['topics']))

            author_data = (k,
                           v['name'],
                           v['affiliations'],
                           len(v['papers']),
                           ';'.join(v['papers']),
                           int(v['citation_num']),
                           ';'.join(v['topics']),
                           ';'.join(v['coauthors'])
                           )

            # Insert query
            query = '''
            INSERT INTO authors (id, name, affiliations, paper_num, papers, citation_num, topics, coauthors)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            '''

            # Execute the query with user data
            cursor.execute(query, author_data)

    conn.commit()

if __name__=='__main__':
    root_dir = '/home/bingxing2/ailab/group/ai4agr/crq/SciSci/OAG'
    start_year = 2010
    end_year = 2020
    year_range = [start_year, end_year]
    least_n_citation = 200
    data_dir = f'/home/bingxing2/ailab/group/ai4agr/crq/SciSci/OAG/data_from_{year_range[0]}to{year_range[1]}_gt_{least_n_citation}_citation'
    if os.path.exists(data_dir) == False:
        os.mkdir(data_dir)

    # connect to database, if it doesn't exist then build one
    conn = sqlite3.connect(os.path.join(data_dir, 'database.db'))
    # create cursor
    cursor = conn.cursor()
    # # 13w papers, 64w authors
    # create_paper_table(conn, cursor, root_dir, data_dir, year_range, least_n_citation)

    # contemporary papers
    year_range = [end_year+1, 9999]
    least_n_citation = 50
    create_contemp_paper_table(conn, cursor, root_dir, year_range, least_n_citation)

    # # extract author info
    # create_author_table(conn, cursor, root_dir, data_dir)
    #
    # # filter author info
    # filter_authors(cursor, data_dir)

    # # fetch all
    # cursor.execute(f"SELECT * FROM papers")
    # all_papers = cursor.fetchall()

    # 关闭游标和连接
    cursor.close()
    conn.close()