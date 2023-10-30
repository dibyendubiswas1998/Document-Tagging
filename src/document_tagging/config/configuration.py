import os
from src.document_tagging.constants import *
from src.document_tagging.entity.config_entity import *
from src.document_tagging.utils.common_utils import read_params



class ConfigManager:
    def __init__(self, secrect_file_path=SECRET_FILE_PATH, config_file_path=CONFIG_FILE_PATH, 
                 params_file_path=PARAMS_FILE_PATH):
        
        self.secrect = read_params(secrect_file_path) # read information from config/secrect.yaml file
        self.config = read_params(config_file_path) # read information from config/config.yaml file
        self.params = read_params(params_file_path) # read information from params.yaml file
    

    def get_log_file_config(self) -> LogConfig:
        try:
            log_file_config = LogConfig(running_log = self.config.logs.log_file)
            return log_file_config # return log file
        
        except Exception as ex:
            raise ex


    def get_data_info_config(self):
        try:
            data_info_config = DataInfo(
                columns = self.secrect.data_info.columns,
                X_feature_name = self.secrect.data_info.X_feature,
                Y_feature_name = self.secrect.data_info.Y_feature,
                log_file = self.config.logs.log_file
            )
            return data_info_config # return DataInfo information

        except Exception as ex:
            raise ex

    
    def get_data_ingestion_config(self):
        try:
            data_ingestion_config = DataIngestionConfig(
                s3_service_name = self.secrect.s3_bucket_access.service_name,
                s3_aws_access_key_id = self.secrect.s3_bucket_access.acess_key,
                s3_aws_secret_access_key = self.secrect.s3_bucket_access.secret_key,
                s3_region_name = self.secrect.s3_bucket_access.region_name,
                s3_bucket_name = self.secrect.s3_bucket_access.bucket_name,
                s3_dataset_1 = self.secrect.s3_bucket_access.dataset_name.df_1,
                s3_dataset_2 = self.secrect.s3_bucket_access.dataset_name.df_2,
                num_records_extract = self.secrect.data_info.no_record_extrcat,
                local_data_file_path = self.config.artifacts.data.data_dir,
                local_data_file_name = self.config.artifacts.data.file_name,
                log_file = self.config.logs.log_file
            )
            return data_ingestion_config # return DataIngestionConfig information

        except Exception as ex:
            raise ex





if __name__ == "__main__":
    pass
