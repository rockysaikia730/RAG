import os
from pathlib import Path
from logger import logging

list_of_files = [
    'RAG/__init__.py',
    'RAG/data_ingestion.py',
    'RAG/embedding.py',
    'RAG/model_api.py',
    'Experiments/experiments.ipynb',
    'StreamlitApp.py',
    'logger.py',
    'exception.py',
    'setup.py'
]

for filepath in list_of_files:
    filepath = Path(filepath)
    filedir,filename = os.path.split(filepath)

    if filedir != "":
        os.makedirs(filedir,exist_ok=True)
        logging.info(f"Creating directory; {filedir} for the file {filename}")

    if((not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0)):
        with open(filepath,"w") as f:
            pass
