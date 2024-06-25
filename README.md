This is a repository for testing open source llms from HunggingFace and deploying them to AWS.

# How to Run LLM Script With Docker

## Prerequsites

### Install GCC

Instance should have installed gcc compiler.

To check if gcc is installed, you can use ```gcc --version```

If it is not installed, you can install with this command: ```sudo apt install gcc```

### Install Make

Instance should have installed Make.

To check if make is installed, you can use ```make --version```

If not, you can install with ```sudo apt install make```

### Install Docker

Instance should have installed Docker runtime to be able to run Docker containers

To check if Docker is installed, you can use ```docker --version```

If it is not installed yet, you can follow this link to install Docker:

https://docs.docker.com/engine/install/ubuntu/#install-using-the-repository

### Install NVIDIA Driver

Instance should have installed NVIDIA driver compatible with your GPU server.

To check if NVIDIA driver is installed, you can run ```nvidia-smi```

If it's not working, you can follow below commands to install driver(ubuntu 22.04).


```
1. wget https://developer.download.nvidia.com/compute/cuda/12.4.1/local_installers/cuda_12.4.1_550.54.15_linux.run

2. sudo sh cuda_12.4.1_550.54.15_linux.run

3. sudo reboot
```

### Install Nvidia Container Toolkit

Instance should have installed Nvidia Container Toolkit so that docker container can access GPU on host computer.

To check if it is installed, you can use ```nvidia-container-cli --version```

If it is not installed yet, you can follow below commands to install.

```
1. curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
&& curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list


2. sudo apt-get update

3. sudo apt-get install -y nvidia-container-toolkit

4. sudo systemctl restart docker (Restart Docker)

```

## How to Run

1. Clone the repository with ```git clone```.

2. Copy ```.env.example``` and rename it as ```.env```

    You should set your HuggingFace access token there.
    To get access token, please follow this link: https://huggingface.co/docs/hub/en/security-tokens

3. Run ```docker-compose up -d --build``` to build image.

4. To embed a new document to vector store, run ```docker exec llm-test python3.10 embed.py --file data/1.txt --classify Yes ``` or update ```run-embed.sh``` and run it.

    Available file formats
    - txt
    - jpg, png, jpeg
    - pdf

    Test files are available inside ```/data/```.


5. To run llm for classification, run ```docker exec llm-test python3.10 main.py --checkpoint bloom --file data/1.txt --output result.csv``` or update ```run-llm.sh``` and run it.

    output: csv file path to save result. Default is result.csv

    Available Checkpoint
    - "bloom"
    - "tinyllama"
    - "llama3"

    Available file formats
    - txt
    - jpg, png, jpeg
    - pdf

    Test files are available inside ```/data/```.
