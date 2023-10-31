import os
import boto3
import pandas as pd
from src.document_tagging.utils.common_utils import log, clean_prev_dirs_if_exis, create_dir
from src.document_tagging.entity.config_entity import DataIngestionConfig


class DataIngestion:
    """
        The DataIngestion class is responsible for loading data from two datasets stored in an S3 bucket, concatenating them
        together, and saving the resulting dataframe to a local directory. It also logs various messages throughout the process.
    """
    def __init__(self, config: DataIngestionConfig) -> None:
        self.config = config
    

    def load_and_save_data(self):
        """
            Load data from two datasets stored in an S3 bucket, concatenate them together, and save the resulting dataframe to a local directory.

            Inputs:
            - self (implicit): The instance of the DataIngestion class.

            Flow:
                1. Retrieve the log file path from the config object.
                2. Create an S3 client using the AWS credentials and region specified in the config object.
                3. Load the first dataset from the S3 bucket using the s3_client and s3_dataset_1 parameters from the config object.
                4. Read the CSV file into a pandas dataframe (df1).
                5. Extract a specified number of records from df1 based on the num_records_extract parameter from the config object.
                6. Load the second dataset from the S3 bucket using the s3_client and s3_dataset_2 parameters from the config object.
                7. Read the CSV file into another pandas dataframe (df2).
                8. Extract a specified number of records from df2 based on the num_records_extract parameter from the config object.
                9. Concatenate df1 and df2 together to create a single dataframe (df).
                10. Clean the local data directory specified in the config object if it already exists.
                11. Create the local data directory specified in the config object.
                12. Save the df dataframe to a CSV file in the local data directory using the local_data_file_name parameter from the config object.
                13. Log various messages throughout the process using the log function.
                14. If any exception occurs, log the error message and raise the exception.

            Outputs:
                Save the data to the specified directory
        """
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

            clean_prev_dirs_if_exis(dir_path=self.config.local_data_directory) # clean the directory if already exists
            create_dir(dirs=[self.config.local_data_directory]) # create the directory for save the data
            log(file_object=log_file, log_message=f"clean and then create the directory {self.config.local_data_directory}") # logs 

            df.to_csv(self.config.local_data_file_name, index=None) # save the data to the local data directory
            log(file_object=log_file, log_message=f"save the data to local directory, {self.config.local_data_directory}") # logs 
        

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
