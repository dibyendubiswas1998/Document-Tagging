from src.document_tagging.config.configuration import ConfigManager
from src.document_tagging.components.data_ingestion import DataIngestion
from src.document_tagging.utils.common_utils import log



STAGE_NAME = "Data Ingestion Config"


class DataIngestionTrainingPipeline:
    """
        The DataIngestionTrainingPipeline class is responsible for executing the data ingestion process. It initializes the 
        ConfigManager class to retrieve the data ingestion configuration, creates an instance of the DataIngestion class 
        with the retrieved configuration, and calls the load_and_save_data method to load data from two datasets stored in 
        an S3 bucket, concatenate them together, and save the resulting dataframe to a local directory.
    """
    def __init__(self):
        pass

    def main(self):
        """
            Executes the data ingestion process.

            This method initializes the ConfigManager class to retrieve the data ingestion configuration,
            creates an instance of the DataIngestion class with the retrieved configuration,
            and calls the load_and_save_data method to load data from two datasets stored in an S3 bucket,
            concatenate them together, and save the resulting dataframe to a local directory.

            Example Usage:
            pipeline = DataIngestionTrainingPipeline()  # Initialize the DataIngestionTrainingPipeline object
            pipeline.main()  # Execute the data ingestion process
        """

        try:
            config_manager = ConfigManager() # ConfigManager class
            data_ingestion_config = config_manager.get_data_ingestion_config() # get data_ingestion_config

            data_ingestion = DataIngestion(config=data_ingestion_config) # data_ingestion object
            data_ingestion.load_and_save_data() # load and save data

        except Exception as ex:
            raise ex
        


if __name__ == "__main__":
    try:
        config_manager = ConfigManager() # ConfigManager class
        log_file = config_manager.get_log_file_config().running_log # get the log file

        log(file_object=log_file, log_message=f"{str('>')*15} Stage: {STAGE_NAME} started {str('<')*15}")
        obj = DataIngestionTrainingPipeline()
        obj.main()
        log(file_object=log_file, log_message=f"{str('>')*15} Stage: {STAGE_NAME} completed {str('<')*15} \n\n")

    except Exception as ex:
        raise ex 
    