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
        """
            Retrieves the log file configuration.

            Returns:
                LogConfig: An instance of the LogConfig class containing the log file path.

            Example Usage:
                # Initialize the ConfigManager object
                config_manager = ConfigManager()

                # Call the get_log_file_config method
                log_file_config = config_manager.get_log_file_config()

                # Access the log file path
                log_file_path = log_file_config.running_log
        """
        try:
            log_file_config = LogConfig(running_log=self.config.logs.log_file)
            return log_file_config

        except Exception as ex:
            raise ex


    def get_data_info_config(self) -> DataInfoConfig:
        """
            Retrieves the data information configuration.

            Returns:
                DataInfoConfig: An instance of the DataInfoConfig class containing the columns, X feature name, Y feature name, and log file path.
        
            Raises:
                Exception: If an error occurs while retrieving the data information configuration.
        """
        try:
            data_info_config = DataInfoConfig(
                columns=self.secrect.data_info.columns,
                X_feature_name=self.secrect.data_info.X_feature,
                Y_feature_name=self.secrect.data_info.Y_feature,
                log_file=self.config.logs.log_file
            )
            return data_info_config

        except Exception as ex:
            raise ex

    
    def get_data_ingestion_config(self) -> DataIngestionConfig:
        """
            Retrieves the data ingestion configuration.

            Returns:
                DataIngestionConfig: An instance of the DataIngestionConfig class containing the data ingestion configuration values.
        
            Raises:
                Exception: If there is an error retrieving the data ingestion configuration.
        """
        try:
            data_ingestion_config = DataIngestionConfig(
                s3_service_name=self.secrect.s3_bucket_access.service_name,
                s3_aws_access_key_id=self.secrect.s3_bucket_access.acess_key,
                s3_aws_secret_access_key=self.secrect.s3_bucket_access.secret_key,
                s3_region_name=self.secrect.s3_bucket_access.region_name,
                s3_bucket_name=self.secrect.s3_bucket_access.bucket_name,
                s3_dataset_1=self.secrect.s3_bucket_access.dataset_name.df_1,
                s3_dataset_2=self.secrect.s3_bucket_access.dataset_name.df_2,
                num_records_extract=self.secrect.data_info.no_record_extrcat,
                local_data_directory=self.config.artifacts.data.data_dir,
                local_data_file_name=self.config.artifacts.data.raw_data_file_name,
                log_file=self.config.logs.log_file
            )
            return data_ingestion_config

        except Exception as ex:
            raise ex


    def get_data_preprocessing_config(self) -> DataPreprocessingConfig:
        """
            Retrieves the data preprocessing configuration.

            Returns:
                DataPreprocessingConfig: An instance of the DataPreprocessingConfig class containing the data preprocessing configuration values.

            Raises:
                Exception: If there is an error retrieving the data preprocessing configuration.
        """
        try:
            data_preprocessing_config = DataPreprocessingConfig(
                data_file_path=self.config.artifacts.data.raw_data_file_name,
                train_torch_file_name=self.config.artifacts.data.train_torch_file_name,
                valid_torch_file_name=self.config.artifacts.data.valid_torch_file_name,
                test_torch_file_name=self.config.artifacts.data.test_torch_file_name,
                json_file=self.config.artifacts.data.num_of_labels_file_name,
                tag2id=self.config.artifacts.data.tag2_file_name,
                columns=self.secrect.data_info.columns,
                X_feature_name=self.secrect.data_info.X_feature,
                Y_feature_name=self.secrect.data_info.Y_feature,
                model_name=self.config.model_info.model_name,
                tokenizer_path=self.config.artifacts.tokenizer.tokenizer_dir,
                split_ratio=self.config.split_ratio.test_size,
                random_state=self.config.split_ratio.random_dtate,
                log_file=self.config.logs.log_file
            )
            return data_preprocessing_config

        except Exception as ex:
            raise ex


    def get_model_training_config(self) -> ModelTrainingConfig:
        """
            Retrieves the model training configuration.

            Returns:
                ModelTrainingConfig: An instance of the ModelTrainingConfig class containing the model training configuration values.
        
            Raises:
                Exception: If there is an error retrieving the model training configuration.
        """
        try:
            model_training_config = ModelTrainingConfig(
                model_name=self.config.model_info.model_name,
                model_log_dir=self.config.artifacts.model.model_log_dir,
                doc_tag_model_dir=self.config.artifacts.model.doc_tag_model_dir,
                tokenizer_file_path=self.config.artifacts.tokenizer.tokenizer_dir,
                train_torch_file_name=self.config.artifacts.data.train_torch_file_name,
                valid_torch_file_name=self.config.artifacts.data.valid_torch_file_name,
                test_torch_file_name=self.config.artifacts.data.test_torch_file_name,
                json_file=self.config.artifacts.data.num_of_labels_file_name,
                tag2id_file=self.config.artifacts.data.tag2_file_name,
                log_file=self.config.logs.log_file,
                num_train_epochs=self.params.TrainingArguments.num_train_epochs,
                warmup_steps=self.params.TrainingArguments.warmup_steps,
                per_device_train_batch_size=self.params.TrainingArguments.per_device_train_batch_size,
                per_device_eval_batch_size=self.params.TrainingArguments.per_device_eval_batch_size,
                learning_rate=self.params.TrainingArguments.learning_rate,
                weight_decay=self.params.TrainingArguments.weight_decay,
                logging_steps=self.params.TrainingArguments.logging_steps,
                evaluation_strategy=self.params.TrainingArguments.evaluation_strategy,
                eval_steps=self.params.TrainingArguments.eval_steps,
                save_steps=self.params.TrainingArguments.save_steps,
                gradient_accumulation_steps=self.params.TrainingArguments.gradient_accumulation_steps,
                save_total_limit=self.params.TrainingArguments.save_total_limit,
                save_strategy=self.params.TrainingArguments.save_strategy
            )
            return model_training_config

        except Exception as ex:
            raise ex


    def get_model_evaluation_config(self) -> ModelEvaluationConfig:
        """
            Retrieves the model evaluation configuration.

            Returns:
                ModelEvaluationConfig: An instance of the ModelEvaluationConfig class containing the model evaluation configuration values.
        """
        try:
            model_evaluation_config = ModelEvaluationConfig(
                train_tensor_data_path=self.config.artifacts.data.train_torch_file_name,
                test_tensor_data_path=self.config.artifacts.data.test_torch_file_name,
                valid_tensor_data_path=self.config.artifacts.data.valid_torch_file_name,
                model_path=self.config.artifacts.model.doc_tag_model_dir,
                tokenizer_path=self.config.artifacts.tokenizer.tokenizer_dir,
                metric_file_path=self.config.artifacts.report.file_name,
                model_log_dir=self.config.artifacts.model.model_log_dir,
                tag2id_file=self.config.artifacts.data.tag2_file_name,
                log_file=self.config.logs.log_file,
                per_device_eval_batch_size=self.params.TrainingArguments.per_device_eval_batch_size,
                mlflow_url=self.secrect.mlflow.MLFLOW_TRACKING_URI,
                all_params=self.params.TrainingArguments
            )
            return model_evaluation_config

        except Exception as ex:
            raise ex


if __name__ == "__main__":
    pass
