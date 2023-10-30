from pathlib import Path
from dataclasses import dataclass
import os


@dataclass(frozen=True)
class LogConfig:
  running_log: Path


@dataclass(frozen=True)
class DataInfo:
  columns: list
  X_feature_name: str
  Y_feature_name: str
  log_file: Path


@dataclass(frozen=True)
class DataIngestionConfig:
  s3_service_name: str
  s3_aws_access_key_id: str
  s3_aws_secret_access_key: str
  s3_region_name: str
  s3_bucket_name: str
  s3_dataset_1: str
  s3_dataset_2: str
  num_records_extract: int
  local_data_file_path: Path
  local_data_file_name: str
  log_file: Path