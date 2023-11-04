from src.document_tagging.config.configuration import ConfigManager
from src.document_tagging.components.model_evaluation_mlflow import ModelEvaluation
from src.document_tagging.utils.common_utils import log



STAGE_NAME = "Model Evaluation"


class ModelEvaluationPipeline:
    """
        The ModelEvaluationPipeline class is responsible for executing the model evaluation process by initializing the 
        necessary objects and calling the evaluation and logging methods.
    """
    def __init__(self):
        pass

    def main(self):
        """
            Executes the model evaluation process by initializing the necessary objects and calling the evaluation and logging methods.

            Example Usage:
                ```
                    pipeline = ModelEvaluationPipeline()
                    pipeline.main()
                ```

            Inputs:
                - None

            Flow:
                1. Create an instance of the `ConfigManager` class.
                2. Retrieve the model evaluation configuration using the `get_model_evaluation_config` method of the `ConfigManager` class.
                3. Create an instance of the `ModelEvaluation` class with the retrieved configuration.
                4. Call the `evaluation` method of the `ModelEvaluation` class to perform the model evaluation.
                5. Call the `log_into_mlflow` method of the `ModelEvaluation` class to log the evaluation results into MLflow.

            Outputs:
                - None
        """
        try:
            config_manager = ConfigManager() # ConfigManager class
            model_evaluation_config = config_manager.get_model_evaluation_config()

            eval = ModelEvaluation(config=model_evaluation_config)
            eval.evaluation()
            eval.log_into_mlflow()

        except Exception as ex:
            raise ex
        


if __name__ == "__main__":
    try:
        config_manager = ConfigManager() # ConfigManager class
        log_file = config_manager.get_log_file_config().running_log # get the log file

        log(file_object=log_file, log_message=f"{str('>')*15} Stage: {STAGE_NAME} started {str('<')*15}")
        obj = ModelEvaluationPipeline()
        obj.main()
        log(file_object=log_file, log_message=f"{str('>')*15} Stage: {STAGE_NAME} completed {str('<')*15} \n\n")

    except Exception as ex:
        raise ex 