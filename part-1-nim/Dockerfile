FROM continuumio/miniconda3:latest

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

COPY environment.yml .
RUN conda env create -f environment.yml && conda clean -afy

SHELL ["conda", "run", "-n", "alchemi-playbook", "/bin/bash", "-c"]
RUN pip install jupyterlab jupyterlab-vim

COPY . .

ENV BGR_SERVER=http://localhost:8000
EXPOSE 8888

CMD ["conda", "run", "--no-capture-output", "-n", "alchemi-playbook", \
     "jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]
