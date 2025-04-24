## Environment
### 1. Clone the Repository
```
git clone https://github.com/RenqiChen/Virtual-Scientists-v2
```
### 2. Create and Activate a Virtual Environment
```
conda create --name virsci python=3.11
conda activate virsci
```
### 3. Install Necessary Packages
Install dependencies of the basic multi-agent framework [CAMEL](https://github.com/camel-ai/camel).
```
cd camel-master
pip install --upgrade pip setuptools
pip install -e .  # This will install dependencies in pyproject.toml and install camel in editable mode
```
Then, install the following necessary packages.
```
pip install ollama
pip install faiss-gpu
```
#### Note
Some other dependencies can be installed as needed.
### 4. Ollama
In our experiments, we use `ollama` to deploy the `llama3.1-8b` and `llama3.1-70b` language models and `mxbai-embed-large` embedding model. The details of deployment could refer to [URL](https://github.com/ollama/ollama-python). Here we show some key steps:

1. Ollama should be installed. The linux version:

```
curl -fsSL https://ollama.com/install.sh | sh
```

2. Run ollama in the path where ollama is installed:

```
./ollama serve
```

3. Pull a model to use with the library:

```
./ollama pull llama3.1
./ollama pull llama3.1:70b
./ollama pull mxbai-embed-large
```

4. Install the ollama python library in your environment:

```
pip install ollama
```

5. Complete the installation and close the terminal.

#### Run Ollama

After pull all models, you need to open the ollama server before running our codes:

```
./ollama serve
```

## Run
### Setup

The raw data is based on the [AMiner Computer Science Dataset](https://www.aminer.cn/aminernetwork) and [Open Academic Graph](https://open.aminer.cn/open/article?id=5965cf249ed5db41ed4f52bf).

After preprocessing, the used data is publicly available at [Google Drive](https://drive.google.com/drive/folders/1asoKTCXtpbQ0DlL5I-z7b5tLut_67d4C?usp=sharing) (Currently we release the preprocessed data of Computer Science Dataset).

* Past paper database is put in the `Papers/papers.tar.gz`, which is used in `paper_folder_path` of Line 50 in `sci_platform/sci_platform_fast.py`. The corresponding embedding database is put in the `Embeddings/faiss_index.index`, which is used in `cpu_index` of Line 258 in `sci_platform/sci_platform_fast.py`.
* Contemporary paper database is put in the `Papers/papers_future.tar.gz`, which is used in `future_paper_folder_path` of Line 51 in `sci_platform/sci_platform_fast.py`. The corresponding embedding database is put in the `Embeddings/faiss_index_future.index`, which is used in `cpu_future_index` of Line 268 in `sci_platform/sci_platform_fast.py`.
* Author knowledge bank is put in the `Authors/books.tar`, which is used in in `input_dir` of Line 49 in `sci_platform/configs/knowledge_config.json` and `author_info_dir` of Line 36 in `sci_platform/sci_platform_fast.py`.
* Adjacency matrix is put in the `adjacency.txt`, which is used in `adjacency_matrix_dir` of Line 37 in `sci_platform/sci_platform_fast.py`.

**Note**

Please replace all paths in `sci_platform/sci_platform_fast.py` with your own settings after download the data.

### Code

Here we explain the roles of several critial files.

* `camel-master/camel/agents/sci_agent.py` defines the customized scientist agent in this project.
* `sci_platform/run.py` is the main execution file.
* `sci_platform/sci_platform_fast.py` defines the platform for the initialization of our multi-agent system.
* `sci_platform/utils/prompt.py` contains all the prompts used.
* `sci_platform/utils/scientist_utils.py` contains all the common functions used.
* `sci_platform/sci_team/SciTeam.py` defines the execution mechanism of each scientist team.

Our code support different collaboration settings. The commonly used arguments:

`--runs`: how many times does the program run

`--team_limit`: the max number of teams for a scientist

`--max_discuss_iteration`: the max discussion iterations for a team in a step

`--max_team_member`: the max team member of a team (including the team leader)

`--epochs`: the allowed time steps for one program run (default value is 6, which is enough for a scientist to finish all steps)

### Single-GPU
```
cd sci_platform
bash port2.sh
```
### Multi-GPU
```
cd sci_platform
bash port1.sh
bash port2.sh
```
