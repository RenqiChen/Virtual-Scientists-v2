<div align="center"><img src="./logo.png" alt="" width="700"></div>

# VirSci-v2
A more powerful version than [Virtual Scientists](https://github.com/RenqiChen/Virtual-Scientists), which supports a million-agent-level scientific collaboration simulation. Our scientific collaboration includes six sections: (1) collaborator selection; (2) topic selection; (3) idea generation; (4) novelty check; (5) abstract generation; (6) review generation.
## 📅 Updates
[2025-05]

1. The related paper is available on [Arxiv](https://arxiv.org/abs/2505.12039).

[2025-05]

1. We release a simple reinforcement learning (RL)-based algorithm for collaborator selection in the `RL-Based` branch.

[2025-04]

1. We release the code and data of VirSci-v2, which is a powerful platform for scientfic collaboration simulation.

## ⚙️ Environment
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

## 💡 Run
### Setup

The raw data is based on the [AMiner Computer Science Dataset](https://www.aminer.cn/aminernetwork) and [Open Academic Graph](https://open.aminer.cn/open/article?id=5965cf249ed5db41ed4f52bf).

After preprocessing, the used data is publicly available at [Google Drive](https://drive.google.com/drive/folders/1ZwWMBQ5oK-l4VuzMa60GbMND0g2EIxIu?usp=sharing).

The files in Google Drive is related to `def _init_` of `class Platform` in `sci_platform/sci_platform_fast.py`.

1. Computer Science Dataset

* Past paper database is put in the `Papers/papers.tar.gz`, which is used in `paper_folder_path`. The corresponding embedding database is put in the `Embeddings/faiss_index.index`, which is used in `paper_index_path`.
* Contemporary paper database is put in the `Papers/papers_future.tar.gz`, which is used in `future_paper_folder_path`. The corresponding embedding database is put in the `Embeddings/faiss_index_future.index`, which is used in `paper_future_index_path`.
* Author knowledge bank is put in the `Authors/books.tar`, which is used in in `input_dir` in `sci_platform/configs/knowledge_config.json` and `author_folder_path`.
* Adjacency matrix is put in the `adjacency.txt`, which is used in `adjacency_matrix_dir`.

2. Open Academic Graph Dataset

* Past paper database is put in the `Papers/papers_OAG.zip`, which is used in `paper_folder_path`. The corresponding embedding database is put in the `Embeddings/faiss_index_OAG.index`, which is used in `paper_index_path`.
* Contemporary paper database is put in the `Papers/papers_future_OAG.tar.gz`, which is used in `future_paper_folder_path`. The corresponding embedding database is put in the `Embeddings/faiss_index_OAG_future.index`, which is used in `paper_future_index_path`.
* Author knowledge bank is put in the `Authors/books_OAG.zip`, which is used in in `input_dir` in `sci_platform/configs/knowledge_config.json` and `author_folder_path`.
* Adjacency matrix is put in the `weight_matrix.txt`, which is used in `adjacency_matrix_dir`.

**Note**

Please replace all paths in `sci_platform/sci_platform_fast.py` with your own settings after download the data.

### Code

Here we explain the roles of several critial files.

* `sci_platform/configs/deploy_config.py` defines all hyper-parameter settings.
* `sci_platform/social_agent/sci_agent.py` defines the customized scientist agent in this project.
* `sci_platform/social_agent/channel.py` defines the message sending and receiving, which is the lowest-level module.
* `sci_platform/inference` controls the messages sent to or received from the channel, which corresponds to different threads.
* `sci_platform/run_fast.py` is the main execution file.
* `sci_platform/sci_platform_fast.py` defines the platform for the initialization of our multi-agent system.
* `sci_platform/utils/prompt.py` contains all the prompts used.
* `sci_platform/utils/scientist_utils.py` contains all the common functions used.
* `sci_platform/sci_team/SciTeam.py` defines the execution mechanism of each scientist team.

Our code support different environment settings. The commonly used arguments in `deploy_config.py`:

1. Deploy Setup

`ips`: the ips for the LLM model deployment

`port`: the ports of the ip for the LLM model deployment

2. Experiment Setup

`agent_num`: how many independent scientists are included in the simulation

`runs`: how many times does the program run

`team_limit`: the max number of teams for a scientist

`max_discuss_iteration`: the max discussion iterations for a team in a step

`max_team_member`: the max team member of a team (including the team leader)

`epochs`: the allowed time steps for one program run (the publish of a complete paper usually needs 5 epochs)

`model_name`: the LLM base model for simulation (e.g., llama3.1)

`leader_mode`: who is the leader (e.g., normal or random)

3. Checkpoint Setup

`checkpoint`: use the checkpoint or create a new program

`test_time`: the name of the test as a checkpoint

`load_time`: the name of the loaded checkpoint

### Distributed Running
#### Single Machine (Single-GPU/Multi-GPU)

In `deploy_config.py`, set the `ips=['127.0.0.1']`. In `port2.sh`, `CUDA_VISIBLE_DEVICES` means how many GPUs are used and `OLLAMA_HOST=0.0.0.0:XXXXX` means the port of one GPU is deployed with a LLM model.

```
cd sci_platform
bash port2.sh
```
### Multi-Machine (Multi-GPU)

In `deploy_config.py`, set the `ips` includes the ips of all machines. `port1.sh` is to deploy LLM models on these distributed machines.

```
cd sci_platform
bash port1.sh
bash port2.sh
```

## 📉 Results and Evaluation
All information during simulation will be saved at `sci_platform/{self.checkpoint_path}/{self.test_time}` folder. In this folder, the structure of data follows:
```
├── citation
├── faiss
├── paper
├── team
├── database_large.db
└── weight_matrix.txt
```

`citation` folder saves the citations of paper which are cited in the nearest epoch.

`faiss` folder saves the embeddings of all paper, including the original paper from public database and newly published paper from the simulation.

`paper` folder saves the text of all paper in the form of `.txt`, including the original paper from public database and newly published paper from the simulation.

`team` folder saves the team data of each scientist.

`database_large.db` saves the data of all paper, including the original paper from public database and newly published paper from the simulation. We use `sqlite3` Python Library to manage the simulation database, where the form of data follows:
```
cursor.execute('''
        CREATE TABLE IF NOT EXISTS papers (
            id INTEGER PRIMARY KEY,
            title TEXT,
            authors TEXT,
            cite_papers TEXT,
            abstract TEXT,
            year INTEGER,
            citation INTEGER,
            reviews TEXT,
            discipline TEXT,
            keyword TEXT
        )
    ''')
``` 
While `title, authors, abstract, keyword` are esay to understand, several other items require explaination: 
* `id` is the ID number of the paper.
* `cite_papers` is the reference of the paper, composed of numbers and combined with comma.
*  `year` is the publication time of the paper. If the paper is extracted from the original public database, the value will be set as `-1`. If the paper is generated from the simulation, the value will set as `epoch` (e.g., 1).
* `citation` is the citation count of the paper.
* `reviews` is a list of review scores, which stores the results of review at different epochs.
* `discipline` denotes the discipline of the paper.

`weight_matrix.txt` saves the collaboration times between scientist after weighted sum.

Please feel free to evaluate these results for specific Science of Science analysis.

## 🙏 Acknowledgements

This project is supported by Shanghai Artificial Intelligence Laboratory.

The multi-agent framework in this work is based on the [CAMEL](https://github.com/camel-ai/camel).

The concurrent distributed system in this work is based on the [OASIS](https://github.com/camel-ai/oasis).

The raw data is based on the [AMiner Computer Science Dataset](https://www.aminer.cn/aminernetwork) and the [Open Academic Graph](https://open.aminer.cn/open/article?id=5965cf249ed5db41ed4f52bf).

## ⚖️ License

This repository is licensed under the [Apache-2.0 License](LICENSE/).

## 📌 Citation
If you find this repository useful, please consider citing our work:
```bibtex
@software{Chen_VirSciv2_2025,
author = {Renqi Chen and Haoyang Su},
month = apr,
title = {{VirSci-v2}},
url = {https://github.com/RenqiChen/Virtual-Scientists-v2},
year = {2025}
}
```
