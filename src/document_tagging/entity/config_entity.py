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