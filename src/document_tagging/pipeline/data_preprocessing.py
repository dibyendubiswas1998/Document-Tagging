from src.document_tagging.config.configuration import ConfigManager
from src.document_tagging.components.data_preprocessing import DataPreprocessing
from src.document_tagging.utils.common_utils import log



STAGE_NAME = "Data Preprocessing"


class DataPreprocessingTrainingPipeline:
    """
        The DataPreprocessingTrainingPipeline class is responsible for executing the data preprocessing pipeline. It initializes the 
        ConfigManager class to retrieve the data preprocessing configuration, creates an instance of the DataPreprocessing class with 
        the configuration, and calls the process method to handle the data preprocessing.
    """
    def __init__(self):
        pass

    def main(self):
        """
            Executes the data preprocessing pipeline.

            This method initializes the ConfigManager class to retrieve the data preprocessing configuration,
            creates an instance of the DataPreprocessing class with the configuration,
            and calls the process method to handle the data preprocessing.

            Example Usage:
            config_manager = ConfigManager()
            data_preprocessing_config = config_manager.get_data_preprocessing_config()

            preprocessing = DataPreprocessing(config=data_preprocessing_config)
            preprocessing.process()
        """

        try:
            config_manager = ConfigManager() # ConfigManager class
            data_preprocessing_config = config_manager.get_data_preprocessing_config()

            preprocessing = DataPreprocessing(config=data_preprocessing_config)
            preprocessing.process()
        
        except Exception as ex:
            raise ex
        


if __name__ == "__main__":
    try:
        config_manager = ConfigManager() # ConfigManager class
        log_file = config_manager.get_log_file_config().running_log # get the log file

        log(file_object=log_file, log_message=f"{str('>')*15} Stage: {STAGE_NAME} started {str('<')*15}")
        obj = DataPreprocessingTrainingPipeline()
        obj.main()
        log(file_object=log_file, log_message=f"{str('>')*15} Stage: {STAGE_NAME} completed {str('<')*15} \n\n")

    except Exception as ex:
        raise ex 
    