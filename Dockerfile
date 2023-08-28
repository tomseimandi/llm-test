FROM python:3.9-slim

WORKDIR /app

# Install git
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install MinIO client
RUN curl https://dl.min.io/client/mc/release/linux-amd64/mc \
  --create-dirs \
  -o $HOME/minio-binaries/mc

RUN chmod +x $HOME/minio-binaries/mc

RUN export PATH=$PATH:$HOME/minio-binaries/

RUN export MC_HOST_s3=https://$AWS_ACCESS_KEY_ID:$AWS_SECRET_ACCESS_KEY@minio.lab.sspcloud.fr

# Download model
RUN mc cp s3/tseimandi/llama-2-7b-chat.ggmlv3.q4_K_M.bin models/llama-2-7b-chat.ggmlv3.q4_K_M.bin

# Clone repository
RUN git clone https://github.com/tomseimandi/llm-test.git .

RUN pip3 install -r requirements.txt

RUN ./install_llama_cublas.sh

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT ["streamlit", "run", "src/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
