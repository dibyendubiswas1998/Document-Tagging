import os
import boto3
import pandas as pd
from src.document_tagging.utils.common_utils import log, clean_prev_dirs_if_exis, create_dir
from src.document_tagging.entity.config_entity import DataIngestionConfig


class DataIngestion:
    def __init__(self, config: DataIngestionConfig) -> None:
        self.config = config
    

    def load_and_save_data(self):
        try:
            log_file = self.config.log_file # mention log file

            # mention S3 bucket configuration:
            s3_client = boto3.resource(
                service_name = self.config.s3_service_name,
                region_name = self.config.s3_region_name,
                aws_access_key_id = self.config.s3_aws_access_key_id,
                aws_secret_access_key = self.config.s3_aws_secret_access_key
            )
            log(file_object=log_file, log_message=f"configure the s3 details") # logs 

            object_1 = s3_client.Bucket(self.config.s3_bucket_name).Object(self.config.s3_dataset_1).get()  # load the object_1
            df1 = pd.read_csv(object_1['Body'])   # load the dataset_1
            df1 = df1[:self.config.num_records_extract] # extract the specific records
            log(file_object=log_file, log_message=f"download the dataset 1 feom s3 bucket, {self.config.s3_dataset_1}") # logs 

            object_2 = s3_client.Bucket(self.config.s3_bucket_name).Object(self.config.s3_dataset_2).get()  # load the object_2
            df2 = pd.read_csv(object_2['Body'])   # load the dataset_2
            df2 = df2[:self.config.num_records_extract] # extract the specific records
            log(file_object=log_file, log_message=f"download the dataset 2 feom s3 bucket, {self.config.s3_dataset_2}") # logs 

            df = pd.concat([df1, df2]) # concat the two datasets together
            log(file_object=log_file, log_message=f"concat the two dataframe together") # logs 

            clean_prev_dirs_if_exis(dir_path=self.config.local_data_file_path) # clean the directory if already exists
            create_dir(dirs=[self.config.local_data_file_path]) # create the directory for save the data
            log(file_object=log_file, log_message=f"clean and then create the directory {self.config.local_data_file_path}") # logs 

            df.to_csv(self.config.local_data_file_name, index=None) # save the data to the local data directory
            log(file_object=log_file, log_message=f"save the data to local directory, {self.config.local_data_file_path}") # logs 
            

        except Exception as ex:
            log_file = self.config.log_file # mention log file
            log(file_object=log_file, log_message=f"cerror will be {ex}") # logs 
            raise ex


if __name__ == "__main__":
    from src.document_tagging.config.configuration import ConfigManager
    config_manager = ConfigManager() # ConfigManager class
    data_ingestion_config = config_manager.get_data_ingestion_config() # get data_ingestion_config

    dd = DataIngestion(data_ingestion_config)
    dd.load_and_save_data()
