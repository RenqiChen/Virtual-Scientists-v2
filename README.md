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
### Note
Some other dependencies can be installed as needed.
## Run
### multi-GPU
```
bash port1.sh
bash port2.sh
```
