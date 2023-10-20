import os
from pathlib import Path
import logging
logging.basicConfig(level=logging.INFO, format="[%(asctime)s]: %(message)s")



# Project Structure:
project_name = "Document Tagging"
template_name = "document_tagging"
list_of_files = [
    "documents/word files/.gitkeep", # all files related to this project
    ".github/workflows/.gitkeep", # CICD pipeline files
    "notebook/.gitkeep", # all the notebook related files

    "artifacts/.gitkeep", # artifacts directory
    "artifacts/data/.gitkeep", # data directory
    "artifacts/model/.gitkeep", # model directory
    "artifacts/tokenizer/.gitkeep", # tokenizer directory
    "artifacts/report/performace_report.json", # artifacts/report directory for performance report
    

    "logs/running_logs.log", # logs file
    
    f"src/{template_name}/__init__.py", # cnnClassifier package

    f"src/{template_name}/components/__init__.py", # component package
    f"src/{template_name}/components/data_ingestion.py", # data-ingestion
    f"src/{template_name}/components/data_preprocessing.py", # data-preprocessing
    f"src/{template_name}/components/model_training.py", # model-training
    f"src/{template_name}/components/model_evaluation_mlflow.py", # model-evaluation-mlflow

    f"src/{template_name}/utils/__init__.py", # utils package
    f"src/{template_name}/utils/common_utils.py", # common_utils module
    # f"src/{template_name}/utils/logging.py", # logging module
    
    f"src/{template_name}/config/__init__.py", # config package 
    f"src/{template_name}/config/configuration.py", # configuration module

    f"src/{template_name}/pipeline/__init__.py", # create a pipeline package for this project
    f"src/{template_name}/pipeline/data_ingestion.py", # stage-01: data-ingestion module
    f"src/{template_name}/pipeline/data_preprocessing.py", # stage-02: data-preprocessing module
    f"src/{template_name}/pipeline/model_training.py", # stage-03: model-training module
    f"src/{template_name}/pipeline/model_evaluation.py", # stage-04: model-evaluation module
    f"src/{template_name}/pipeline/prediction.py", # prediction based on user's data

    f"src/{template_name}/entity/__init__.py", # entity package
    f"src/{template_name}/entity/config_entity.py", # config-entity module

    f"src/{template_name}/constants/__init__.py", # constants package


    "config/config.yaml", # config/config.yaml file
    "config/secrect.yaml", # config/secrect.yaml file
    "params.yaml", # params.yaml file
    "dvc.yaml", # dvc.yaml file    
    
    "templates/index.html", # index.html file
    "static/css/style.css", # static/css/style.css file
    "static/js/script.js", # static/js/script.js file
    "static/image/.gitkeep", # static/image directory

    "main.py", # main.py file
    "setup.py", # setup.py file
    "app.py", # app.py file
    "requirements.txt", # requirements.txt file    

    "Dockerfile", # Dockerfile
    "deployment.yaml", # deployment.yaml file for kubernetest
    "services.yaml", # services.yaml file for Kubernetes
]


def create_project_template(project_template_lst):
    """
    Creates directories and files based on the provided file paths.

    Args:
        project_template_lst (list): A list of file paths.

    Returns:
        None

    Raises:
        OSError: If there is an error creating directories or files.
        IOError: If there is an error creating directories or files.
        Exception: If there is an unknown error.

    Example Usage:
        project_template_lst = ['dir1/file1.txt', 'dir2/file2.txt', 'file3.txt']
        create_project_template(project_template_lst)
    """
    try:
        for filepath in project_template_lst:
            filepath = Path(filepath)
            file_dir, file_name = filepath.parent, filepath.name

            if file_dir != "":
                Path(file_dir).mkdir(parents=True, exist_ok=True)
                logging.info(f"Created directory: {file_dir}")

            if (not filepath.exists()) or (filepath.stat().st_size == 0):
                filepath.touch()
                logging.info(f"Created file: {filepath}")
            else:
                logging.info(f"{file_name} already exists")

    except (OSError, IOError) as e:
        logging.error(f"Error: {e}")
    except Exception as e:
        logging.error(f"Unknown error: {e}")
        



if __name__ == "__main__":
    logging.info(f"Created project template for: {project_name}")
    create_project_template(list_of_files)