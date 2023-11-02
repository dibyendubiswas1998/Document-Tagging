from pathlib import Path
from dataclasses import dataclass
import os


@dataclass(frozen=True)
class LogConfig:
    """
      Represents the configuration for logging in a Python application.

      Attributes:
          running_log (Path): The path to the running log file.
    """
    running_log: Path


@dataclass(frozen=True)
class DataInfoConfig:
    """
      A data class that represents the configuration for data information.

      Attributes:
          columns (list): A list that represents the columns of the data.
          X_feature_name (str): A string that represents the name of the X feature.
          Y_feature_name (str): A string that represents the name of the Y feature.
          log_file (Path): A `Path` object that represents the path to the log file.
    """
    columns: list
    X_feature_name: str
    Y_feature_name: str
    log_file: Path


@dataclass(frozen=True)
class DataIngestionConfig:
    """
      Represents the configuration for data ingestion.

      Attributes:
          s3_service_name (str): The name of the S3 service.
          s3_aws_access_key_id (str): The AWS access key ID for the S3 service.
          s3_aws_secret_access_key (str): The AWS secret access key for the S3 service.
          s3_region_name (str): The region name for the S3 service.
          s3_bucket_name (str): The name of the S3 bucket.
          s3_dataset_1 (str): The name of the first dataset in the S3 bucket.
          s3_dataset_2 (str): The name of the second dataset in the S3 bucket.
          num_records_extract (int): The number of records to extract.
          local_data_directory (Path): The local data directory.
          local_data_file_name (Path): The name of the local data file.
          log_file (Path): The file path for logging.
    """
    s3_service_name: str
    s3_aws_access_key_id: str
    s3_aws_secret_access_key: str
    s3_region_name: str
    s3_bucket_name: str
    s3_dataset_1: str
    s3_dataset_2: str
    num_records_extract: int
    local_data_directory: Path
    local_data_file_name: Path
    log_file: Path



@dataclass(frozen=True)
class DataPreprocessingConfig:
    """
      A data class that represents the configuration for data preprocessing.

      Attributes:
          data_file_path (Path): The path to the data file.
          train_torch_file_name (Path): The path to the training data file in Torch format.
          valid_torch_file_name (Path): The path to the validation data file in Torch format.
          test_torch_file_name (Path): The path to the test data file in Torch format.
          json_file (Path): The path to the JSON file.
          tag2id (Path): The path to the JSON file.
          columns (list): The list of column names in the data file.
          X_feature_name (str): The name of the input feature.
          Y_feature_name (str): The name of the target feature.
          model_name (str): The name of the model.
          tokenizer_path (Path): The path to the tokenizer.
          split_ratio (float): The ratio for splitting the data into training and validation sets.
          random_state (int): The random state for reproducibility.
          log_file (Path): The path to the log file.
    """
    data_file_path: Path
    train_torch_file_name: Path
    valid_torch_file_name: Path
    test_torch_file_name: Path
    json_file: Path
    tag2id: Path
    columns: list
    X_feature_name: str
    Y_feature_name: str
    model_name: str
    tokenizer_path: Path
    split_ratio: float
    random_state: int
    log_file: Path


@dataclass(frozen=True)
class ModelTrainingConfig:
    """
        Represents the configuration for model training.

        Fields:
            - model_name: The name of the model to be trained (string)
            - model_dir: The file path to save the model
            - tokenizer_file_path: The file path to the tokenizer used for tokenizing the input data (Path object)
            - train_torch_file_name: The file name of the training data in torch format (Path object)
            - valid_torch_file_name: The file name of the validation data in torch format (Path object)
            - test_torch_file_name: The file name of the test data in torch format (Path object)
            - json_file: The file path to the JSON configuration file (Path object)
            - tag2id_file: The file path to the tag-to-id mapping file (Path object)
            - log_file: The file path to the log file for training (Path object)
            - num_train_epochs: The number of training epochs (integer)
            - warmup_steps: The number of warmup steps for learning rate scheduling (integer)
            - per_device_train_batch_size: The batch size for training per device (integer)
            - per_device_eval_batch_size: The batch size for evaluation per device (integer)
            - learning_rate: The learning rate for optimization (float)
            - weight_decay: The weight decay for optimization (float)
            - logging_steps: The number of steps between logging (integer)
            - evaluation_strategy: The strategy for evaluation (string)
            - eval_steps: The number of steps between evaluation (integer)
            - save_steps: The number of steps between model saving (integer)
            - gradient_accumulation_steps: The number of steps for gradient accumulation (integer)
            - save_total_limit: The maximum number of saved models (integer)
            - save_strategy: The strategy for model saving (string)
    """
    model_name: str
    model_log_dir: Path
    doc_tag_model_dir: Path
    tokenizer_file_path: Path
    train_torch_file_name: Path
    valid_torch_file_name: Path
    test_torch_file_name: Path
    json_file: Path
    tag2id_file: Path
    log_file: Path
    num_train_epochs: int
    warmup_steps: int
    per_device_train_batch_size: int
    per_device_eval_batch_size: int
    learning_rate: str
    weight_decay: float
    logging_steps: int
    evaluation_strategy: str
    eval_steps: int
    save_steps: int
    gradient_accumulation_steps: int
    save_total_limit: int
    save_strategy: str