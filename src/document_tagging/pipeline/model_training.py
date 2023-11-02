from src.document_tagging.config.configuration import ConfigManager
from src.document_tagging.components.model_training import ModelTraining
from src.document_tagging.utils.common_utils import log



STAGE_NAME = "Model Training"


class ModelTrainingPipeline:
    """
        The ModelTrainingPipeline class is responsible for initializing the necessary objects and starting the model training 
        process.
    """

    def __init__(self):
        pass

    def main(self):
        """
            The main method is responsible for initializing the necessary objects and starting the model training process.

            Example Usage:
            config_manager = ConfigManager()
            model_training_config = config_manager.get_model_training_config()

            train = ModelTraining(config=model_training_config)
            train.train_model()

            Inputs: None
            Outputs: None
        """

        try:
            config_manager = ConfigManager() # ConfigManager class
            model_training_config = config_manager.get_model_training_config()

            train = ModelTraining(config=model_training_config)
            train.train_model()
        
        except Exception as ex:
            raise ex
        


if __name__ == "__main__":
    try:
        config_manager = ConfigManager() # ConfigManager class
        log_file = config_manager.get_log_file_config().running_log # get the log file

        log(file_object=log_file, log_message=f"{str('>')*15} Stage: {STAGE_NAME} started {str('<')*15}")
        obj = ModelTrainingPipeline()
        obj.main()
        log(file_object=log_file, log_message=f"{str('>')*15} Stage: {STAGE_NAME} completed {str('<')*15} \n\n")

    except Exception as ex:
        raise ex 