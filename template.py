import os
from pathlib import Path
import logging

#setting logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

#define project name
project_name="text_summarizer"

#define list of files to be created
list_of_files = [
    ".github/workflows/.gitkeep",
    f"src/{project_name}/__init__.py",
    f"src/{project_name}/components/__init__.py",
    f"src/{project_name}/utils/__init__.py",
    f"src/{project_name}/utils/common.py",
    f"src/{project_name}/logging/__init__.py",
    f"src/{project_name}/config/__init__.py",
    f"src/{project_name}/config/configuration.py",
    f"src/{project_name}/pipeline/_init_.py",
    f"src/{project_name}/entity/_init_.py",
    f"src/{project_name}/constants/_init_.py",
    "config/config.yaml",
    "params.yaml",
    "app.py",
    "main.py",
    "setup.py",
    "notebooks/eda.py",
    "requirements.txt"
]

for file_path in list_of_files:
    file_path = Path(file_path)
    file_dir, file_name = os.path.split(file_path)
    if file_dir:
        os.makedirs(file_dir, exist_ok=True)
    
    #if file does not exist, create an empty file
    if not file_path.exists():
        with open(file_path, "w") as f:
            f.write("")
        logging.info(f"Created {file_path}")
    else:
        logging.warning(f"{file_path} already exists")  
        
        