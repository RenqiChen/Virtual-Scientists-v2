# =========== Copyright 2023 @ CAMEL-AI.org. All Rights Reserved. ===========
# Licensed under the Apache License, Version 2.0 (the “License”);
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an “AS IS” BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========== Copyright 2023 @ CAMEL-AI.org. All Rights Reserved. ===========
import subprocess
import threading
import time
import os
import requests
import deploy_config

def check_port_open(host, port):
    while True:
        url = f"http://{host}:{port}/health"
        try:
            response = requests.get(url)
            if response.status_code == 200:
                break
            else:
                time.sleep(0.3)
        except Exception:
            time.sleep(0.3)


if __name__ == "__main__":
    cwd = os.getcwd()

    port_list = deploy_config.port
    ports = [port_list[i:i+deploy_config.port4GPU] for i in range(0, len(port_list), deploy_config.port4GPU)]
    gpus = [0]

    all_ports = [port for i in gpus for port in ports[i]]
    print("All ports: ", all_ports, '\n\n')

    t = None
    for ip in deploy_config.ips:
        for i in range(3):
            for j, gpu in enumerate(gpus):
                if deploy_config.use_ollama:
                    cwd = deploy_config.ollama_dir
                    cmd = (
                        f"CUDA_VISIBLE_DEVICES={gpu} OLLAMA_HOST={ip}:{ports[j][i]} ./ollama serve &")
                else:
                    cmd = (
                        f"CUDA_VISIBLE_DEVICES={gpu} python -m "
                        f"vllm.entrypoints.openai.api_server --model "
                        f"'/ibex/user/yangz0h/open_source_llm/llama-3' "
                        f"--served-model-name 'llama-3' "
                        f"--host {host} --port {ports[j][i]} --gpu-memory-utilization "
                        f"0.3 --disable-log-stats")
                t = threading.Thread(target=subprocess.run,
                                     args=(cmd, ),
                                     kwargs={"shell": True, "cwd": cwd},
                                     daemon=True)
                t.start()
            # check_port_open(host, ports[0][i])

    t.join()