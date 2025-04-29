# VirSci-v2
A more powerful version than [Virtual Scientists](https://github.com/RenqiChen/Virtual-Scientists), which supports a million-agent-level scientific collaboration simulation.
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

## Run
### Setup

The raw data is based on the [AMiner Computer Science Dataset](https://www.aminer.cn/aminernetwork) and [Open Academic Graph](https://open.aminer.cn/open/article?id=5965cf249ed5db41ed4f52bf).

After preprocessing, the used data is publicly available at [Google Drive](https://drive.google.com/drive/folders/1asoKTCXtpbQ0DlL5I-z7b5tLut_67d4C?usp=sharing) (Currently we release the preprocessed data of Computer Science Dataset).

* Past paper database is put in the `Papers/papers.tar.gz`, which is used in `paper_folder_path` of Line 48 in `sci_platform/sci_platform_fast.py`. The corresponding embedding database is put in the `Embeddings/faiss_index.index`, which is used in `paper_index_path` of Line 51 in `sci_platform/sci_platform_fast.py`.
* Contemporary paper database is put in the `Papers/papers_future.tar.gz`, which is used in `future_paper_folder_path` of Line 49 in `sci_platform/sci_platform_fast.py`. The corresponding embedding database is put in the `Embeddings/faiss_index_future.index`, which is used in `paper_future_index_path` of Line 268 in `sci_platform/sci_platform_fast.py`.
* Author knowledge bank is put in the `Authors/books.tar`, which is used in in `input_dir` of Line 13 in `sci_platform/configs/knowledge_config.json` and `author_folder_path` of Line 47 in `sci_platform/sci_platform_fast.py`.
* Adjacency matrix is put in the `adjacency.txt`, which is used in `adjacency_matrix_dir` of Line 50 in `sci_platform/sci_platform_fast.py`.

**Note**

Please replace all paths in `sci_platform/sci_platform_fast.py` with your own settings after download the data.

### Code

Here we explain the roles of several critial files.

* `sci_platform/configs/deploy_config.py` defines all hyper-parameter settings.
* `camel-master/camel/agents/sci_agent.py` defines the customized scientist agent in this project.
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

## Acknowledgements

This project is supported by Shanghai Artificial Intelligence Laboratory.

The multi-agent framework in this work is based on the [CAMEL](https://github.com/camel-ai/camel).

The concurrent distributed system in this work is based on the [OASIS](https://github.com/camel-ai/oasis).

The raw data is based on the [AMiner Computer Science Dataset](https://www.aminer.cn/aminernetwork) and the [Open Academic Graph](https://open.aminer.cn/open/article?id=5965cf249ed5db41ed4f52bf).

## License

This repository is licensed under the [Apache-2.0 License](LICENSE/).
