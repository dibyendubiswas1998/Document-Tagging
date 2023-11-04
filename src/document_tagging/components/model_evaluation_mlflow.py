from src.document_tagging.utils.common_utils import log, load_json_file, load_torch_data, save_json_file
from src.document_tagging.entity.config_entity import ModelEvaluationConfig
from transformers import AutoModelForTokenClassification, AutoTokenizer
from transformers import Trainer, TrainingArguments
import torch
import datasets
import numpy as np
import mlflow
import mlflow.pytorch
from urllib.parse import urlparse
import warnings
warnings.filterwarnings("ignore")




class ModelEvaluation:
    """
        The `ModelEvaluation` class is responsible for evaluating a token classification model using test and validation datasets. It computes evaluation metrics such as precision, recall, F1 score, and accuracy. It also logs the evaluation results and parameters into MLflow, a platform for managing the machine learning lifecycle.

        Example Usage:
            config = ModelEvaluationConfig()
            evaluation = ModelEvaluation(config)
            evaluation.evaluation()
            evaluation.log_into_mlflow()

        Methods:
            __init__(self, config: ModelEvaluationConfig) -> None:
                Initializes the `ModelEvaluation` class with a `ModelEvaluationConfig` object.

            evaluation(self):
                Evaluates the token classification model using test and validation datasets. This method computes evaluation metrics such as precision, recall, F1 score, and accuracy. It uses the `Trainer` class from the `transformers` library to perform the evaluation.

            log_into_mlflow(self):
                Logs the evaluation metrics and model parameters into MLflow, a platform for managing the machine learning lifecycle. It also saves the model in the MLflow model registry if the tracking URI is not a file store.

        Fields:
            config: An instance of the `ModelEvaluationConfig` class that holds the configuration parameters for model evaluation.
            log_file: The path to the log file.
            device: The device (CPU or GPU) on which the model will be evaluated.
            model: The token classification model.
            tokenizer: The tokenizer used for tokenizing the input data.
            metric: The evaluation metric used for computing precision, recall, F1 score, and accuracy.
            test_dataset: The test dataset used for evaluation.
            valid_dataset: The validation dataset used for evaluation.
            tag2id: A dictionary mapping tags to their corresponding IDs.
            label_list: A list of labels used for evaluation.
            training_args: The training arguments for the `Trainer` object.
            trainer: The `Trainer` object used for evaluation.
            test_result: The evaluation results on the test dataset.
            validation_result: The evaluation results on the validation dataset.
            performance_report: A dictionary containing the evaluation parameters.
            model: The token classification model used for logging into MLflow.
            report_file: The evaluation report loaded from a JSON file.
            metrics_test: The evaluation metrics for the test dataset.
            metrics_valid: The evaluation metrics for the validation dataset.
    """
    
    def __init__(self, config: ModelEvaluationConfig) -> None:
        self.config = config
    
    def evaluation(self):
        """
            Evaluates a token classification model using test and validation datasets.

            This method computes evaluation metrics such as precision, recall, F1 score, and accuracy. It uses the `Trainer` class from the `transformers` library to perform the evaluation.

            Example Usage:
            ```python
                config = ModelEvaluationConfig()
                evaluation = ModelEvaluation(config)
                evaluation.evaluation()
            ```

            Inputs:
            - self (ModelEvaluation): The instance of the `ModelEvaluation` class.

            Flow:
                1. Load the model and tokenizer using the provided paths.
                2. Load the test and validation datasets.
                3. Define a function `compute_metrics` to compute evaluation metrics.
                4. Create a `Trainer` object with the loaded model, tokenizer, and the `compute_metrics` function.
                5. Evaluate the model on the test dataset and store the results.
                6. Evaluate the model on the validation dataset and store the results.
                7. Save the evaluation parameters to a JSON file.
                8. Log the evaluation results and parameters.

            Outputs:
                None
        """
        try:
            log_file = self.config.log_file # mention log file
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = AutoModelForTokenClassification.from_pretrained(self.config.model_path).to(device) # load the model
            tokenizer = AutoTokenizer.from_pretrained(self.config.tokenizer_path) # load the tokenizer
            metric = datasets.load_metric("seqeval") # loads the seqeval metric

            train_dataset = load_torch_data(self.config.train_tensor_data_path) # loads the train dataset
            test_dataset = load_torch_data(self.config.test_tensor_data_path) # load the test dataset
            vaild_dataset = load_torch_data(self.config.test_tensor_data_path) # load the vaild dataset

            tag2id = load_json_file(self.config.tag2id_file)
            label_list = [key for key, val in tag2id.items()]

            def compute_metrics(eval_preds):
                """
                    Computes evaluation metrics.

                    This function is used to compute evaluation metrics such as precision, recall, F1 score, and accuracy.

                    Args:
                    - eval_preds (tuple): A tuple containing the predicted logits and labels.

                    Returns:
                    - dict: A dictionary containing the computed metrics (precision, recall, F1 score, and accuracy).
                """
                pred_logits, labels = eval_preds
                pred_logits = np.argmax(pred_logits, axis=2)
                # the logits and the probabilities are in the same order,
                # so we don’t need to apply the softmax
                # We remove all the values where the label is -100
                predictions = [
                    [label_list[eval_preds] for (eval_preds, l) in zip(prediction, label) if l != -100] for prediction, label in zip(pred_logits, labels)
                ]

                true_labels = [
                    [label_list[l] for (eval_preds, l) in zip(prediction, label) if l != -100] for prediction, label in zip(pred_logits, labels)
                ]
                results = metric.compute(predictions=predictions, references=true_labels)
                return {
                    "precision": results["overall_precision"],
                    "recall": results["overall_recall"],
                    "f1": results["overall_f1"],
                    "accuracy": results["overall_accuracy"],
                }

            # Create a Trainer object for evaluation
            training_args = TrainingArguments(
                output_dir=self.config.model_log_dir,
                per_device_eval_batch_size=self.config.per_device_eval_batch_size,
            )
            trainer = Trainer(
                model=model,
                args=training_args,
                tokenizer=tokenizer,
                compute_metrics=compute_metrics,
            )

            # train_result = trainer.evaluate(train_dataset) # evaluate based on training dataset
            test_result = trainer.evaluate(test_dataset) # evaluate based on test dataset
            # validation_result = trainer.evaluate(vaild_dataset) # evaluate based on vaild dataset

            performance_report = {
                # "train_validation_params": train_result,
                "test_validation_params": test_result,
                # "valid_validation_params": validation_result
            }

            save_json_file(file_path=self.config.metric_file_path, report=performance_report) # save the evaluation parameters
            log(file_object=log_file, log_message=f"evaluate the model based on train and test and valid dataset and save into {self.config.metric_file_path}") #logs
            log(file_object=log_file, log_message=f"test result: {test_result}") # logs
            # log(file_object=log_file, log_message=f"valid result: {validation_result}") # logs

        except Exception as ex:
            log_file = self.config.log_file # mention log file
            log(file_object=log_file, log_message=f"error will be: {ex}") # logs
            raise ex


    def log_into_mlflow(self):
        """
            The log_into_mlflow method logs the evaluation metrics and model parameters into MLflow, a platform for managing the
            machine learning lifecycle. It also saves the model in the MLflow model registry if the tracking URI is not a file store.
        """
        try:
            log_file = self.config.log_file # mention log file
            model = AutoModelForTokenClassification.from_pretrained(self.config.model_path) # load the model
            
            report_file = load_json_file(self.config.metric_file_path) # loads the report.jso file
            # metrics_train = {
            #     "train_eval_loss": report_file["train_validation_params"]["eval_loss"],
            #     "train_eval_precision": report_file["train_validation_params"]["eval_precision"],
            #     "train_eval_recall": report_file["train_validation_params"]["eval_recall"],
            #     "train_eval_f1": report_file["train_validation_params"]["eval_f1"],
            #     "train_eval_accuracy": report_file["train_validation_params"]["eval_accuracy"],
            #     "train_eval_runtime": report_file["train_validation_params"]["eval_runtime"],
            #     "train_eval_samples_per_second": report_file["train_validation_params"]["eval_samples_per_second"],
            #     "train_eval_steps_per_second": report_file["train_validation_params"]["eval_steps_per_second"]
            # }
            metrics_test = {
                "test_eval_loss": report_file["test_validation_params"]["eval_loss"],
                "test_eval_precision": report_file["test_validation_params"]["eval_precision"],
                "test_eval_recall": report_file["test_validation_params"]["eval_recall"],
                "test_eval_f1": report_file["test_validation_params"]["eval_f1"],
                "test_eval_accuracy": report_file["test_validation_params"]["eval_accuracy"],
                "test_eval_runtime": report_file["test_validation_params"]["eval_runtime"],
                "test_eval_samples_per_second": report_file["test_validation_params"]["eval_samples_per_second"],
                "test_eval_steps_per_second": report_file["test_validation_params"]["eval_steps_per_second"]
            }
            # metrics_valid = {
            #     "valid_eval_loss": report_file["valid_validation_params"]["eval_loss"],
            #     "valid_eval_precision": report_file["valid_validation_params"]["eval_precision"],
            #     "valid_eval_recall": report_file["valid_validation_params"]["eval_recall"],
            #     "valid_eval_f1": report_file["valid_validation_params"]["eval_f1"],
            #     "valid_eval_accuracy": report_file["valid_validation_params"]["eval_accuracy"],
            #     "valid_eval_runtime": report_file["valid_validation_params"]["eval_runtime"],
            #     "valid_eval_samples_per_second": report_file["valid_validation_params"]["eval_samples_per_second"],
            #     "valid_eval_steps_per_second": report_file["valid_validation_params"]["eval_steps_per_second"]
            # }
            
            mlflow.set_registry_uri(self.config.mlflow_url)
            mlflow.set_tracking_uri(self.config.mlflow_url)
            tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

            with mlflow.start_run():
                mlflow.log_params(self.config.all_params) # logs the TrainingArguments params
                # mlflow.log_metrics(metrics_train) # logs the Metrics based on train_dataset
                mlflow.log_metrics(metrics_test) # logs the Metrics based on test_dataset
                # mlflow.log_metrics(metrics_valid) # logs the Metrics based on valid_dataset
                
                # Model registry does not work with file store
                if tracking_url_type_store != "file":
                    mlflow.pytorch.log_model(model, "model", registered_model_name="doc_tag")
                else:
                    mlflow.pytorch.log_model(model, "model")

            log(file_object=log_file, log_message=f"logs into mlfow dagshub") # logs

        except Exception as ex:
            log_file = self.config.log_file # mention log file
            log(file_object=log_file, log_message=f"error will be: {ex}") # logs
            raise ex



if __name__ == "__main__":
    from src.document_tagging.config.configuration import ConfigManager
    config_manager = ConfigManager()
    model_evaluation_config = config_manager.get_model_evaluation_config()

    eval = ModelEvaluation(config=model_evaluation_config)
    eval.evaluation()
    eval.log_into_mlflow()
