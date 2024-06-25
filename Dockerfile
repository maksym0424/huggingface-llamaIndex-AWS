# Use NVIDIA's CUDA enabled Ubuntu base image
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

# Set the working directory in the container
WORKDIR /app

# Install Python 3.10 and other dependencies
RUN apt-get update && \
    apt-get install -y software-properties-common wget && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y python3.10 python3.10-distutils libcudnn8 libcudnn8-dev && \
    wget https://bootstrap.pypa.io/get-pip.py && \
    apt-get update && \
    apt-get install -y ffmpeg && \
    apt-get install -y tesseract-ocr tesseract-ocr-eng libtesseract-dev poppler-utils && \
    python3.10 get-pip.py && \
    rm get-pip.py && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*
    
# Copy the requirements.txt file into the working directory
COPY requirements.txt .

# Install Python requirements
RUN python3.10 -m pip install --no-cache-dir -r requirements.txt

# Copy the entire backend application to the container
COPY . .

# Expose the port your application runs on
EXPOSE 8000

CMD ["tail", "-f", "/dev/null"]
# Specify the custom entrypoint script
# CMD ["python3.10", "main.py"]

# CMD can be used to provide default arguments to the ENTRYPOINT if needed
# CMD ["arg1", "arg2"]  # Uncomment and modify this line if arguments are required