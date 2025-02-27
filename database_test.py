import json
import re
from tqdm import tqdm
import os
import numpy as np
import sqlite3

root_dir = '/home/bingxing2/ailab/scxlab0066/SocialScience/database'
# load database
global_database_name = 'database_large.db'
global_conn = sqlite3.connect(os.path.join(root_dir, global_database_name))
global_cursor = global_conn.cursor()
# load all papers
paper_list = global_cursor.execute('SELECT * FROM papers').fetchall()
i=0
# save txt
print(len(paper_list))
paper = paper_list[(len(paper_list)-1)]
id = paper[0]
title = paper[1]
authors = paper[2]
affiliations = paper[3]
year = paper[4]
reviews = paper[7]
print(reviews)
# close
global_cursor.close()
global_conn.close()
