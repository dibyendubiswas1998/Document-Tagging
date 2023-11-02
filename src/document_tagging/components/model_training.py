from src.document_tagging.utils.common_utils import log, load_json_file, load_torch_data, clean_prev_dirs_if_exis, create_dir
from src.document_tagging.entity.config_entity import ModelTrainingConfig
from transformers import AutoModelForTokenClassification, AutoTokenizer
import json
import numpy as np
import torch
import datasets
from transformers import Trainer, TrainingArguments
import warnings
warnings.filterwarnings("ignore")



class ModelTraining:
    """
        The `ModelTraining` class is responsible for training a token classification model using the Hugging Face Transformers library. It loads a pre-trained model, tokenizer, and training data, sets up the training arguments, and trains the model. It also computes evaluation metrics during training and saves the trained model.

        Example Usage:
            # Initialize the ModelTrainingConfig object
            config = ModelTrainingConfig()

            # Initialize the ModelTraining class object
            model_training = ModelTraining(config)

            # Train the model
            model_training.train_model()

        Methods:
            - __init__(self, config: ModelTrainingConfig): Initializes the `ModelTraining` class with a `ModelTrainingConfig` object.
            - load_model(self): Loads a pre-trained model for token classification.
            - data_collector(self, features): Collects and organizes the input features for training the model.
            - train_model(self): Trains a token classification model using the Hugging Face Transformers library.

        Fields:
            - config: An instance of `ModelTrainingConfig` that holds the configuration parameters for training the model.
    """
    def __init__(self, config: ModelTrainingConfig) -> None:
        self.config = config

    def load_model(self):
        """
            Loads a pre-trained model for token classification.

            Returns:
                model (AutoModelForTokenClassification): The loaded pre-trained model for token classification.

            Raises:
                Exception: If there is an error loading the model.
        """
        try:
            log_file = self.config.log_file # mention log file
            dict = load_json_file(self.config.json_file) # load the number of labels
            device = "cuda" if torch.cuda.is_available() else "cpu" # check which one available: cpu or cuda (gpu)
            model = AutoModelForTokenClassification.from_pretrained(self.config.model_name, num_labels=dict["no_labels"]).to(device) # load the model
            log(file_object=log_file, log_message=f"load the model, {self.config.model_name}") # logs
            return model # return the model

        except Exception as ex:
            log_file = self.config.log_file # mention log file
            log(file_object=log_file, log_message=f"error will be: {ex}") # logs
            raise ex


    def data_collector(self, features):
        """
            Collects and organizes the input features for training the model.

            Args:
                features (list): A list of dictionaries representing the input features. Each dictionary contains the keys 'input_ids', 'attention_mask', and 'labels', which correspond to the input IDs, attention mask, and labels for each feature.

            Returns:
                dict: A dictionary containing the padded input IDs, attention mask, and labels. The keys are 'input_ids', 'attention_mask', and 'labels', respectively.

            Raises:
                Exception: If an error occurs during the data collection process.

            Example Usage:
                # Initialize the ModelCreation class object
                model_creation = ModelCreation()

                # Define the input features
                features = [
                    {'input_ids': [1, 2, 3], 'attention_mask': [1, 1, 1], 'labels': [0, 1, 0]},
                    {'input_ids': [4, 5, 6], 'attention_mask': [1, 1, 1], 'labels': [1, 0, 1]}
                ]

                # Call the data_collector method
                data = model_creation.data_collector(features)
        """
        try:
            input_ids = torch.nn.utils.rnn.pad_sequence([feature['input_ids'] for feature in features], batch_first=True)
            attention_mask = torch.nn.utils.rnn.pad_sequence([feature['attention_mask'] for feature in features], batch_first=True)
            labels = torch.nn.utils.rnn.pad_sequence([feature['labels'] for feature in features], batch_first=True)
            return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

        except Exception as ex:
            log_file = self.config.log_file # mention log file
            log(file_object=log_file, log_message=f"error will be: {ex}") # logs
            raise ex


    def train_model(self):
        """
            Trains a token classification model using the Hugging Face Transformers library.

            This method loads the pre-trained model, tokenizer, and training data. It sets up the training arguments and trains the model. 
            It also computes evaluation metrics during training and saves the trained model.

            Inputs:
            - config (ModelTrainingConfig): An object that holds the configuration parameters for training the model.

            Outputs:
            - None. The method trains the model and saves it, but does not return any output.
        """
        try:
            log_file = self.config.log_file # mention log file
            metric = datasets.load_metric("seqeval") # loads the seqeval metric
            model = self.load_model()
            tokenizer = AutoTokenizer.from_pretrained(self.config.tokenizer_file_path) # loads the tokenizer

            # create the compute_metrics function for compute the metrics during training
            tag2id = load_json_file(file_path=self.config.tag2id_file) # loads the tag2id
            label_list = [key for key, val in tag2id.items()]
        
            def compute_metrics(eval_preds):
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

            training_dataset = load_torch_data(torch_data_file_path=self.config.train_torch_file_name) # load trainings dataset
            validation_dataset = load_torch_data(torch_data_file_path=self.config.valid_torch_file_name) # load validation dataset
      
            # clean & create directories:
            clean_prev_dirs_if_exis(dir_path=self.config.model_log_dir) # remove old directory if it exists
            clean_prev_dirs_if_exis(dir_path=self.config.doc_tag_model_dir) # remove old directory if it exists
            create_dir(dirs=[self.config.model_log_dir, self.config.doc_tag_model_dir]) # create directories

            # provide the training arguments params:
            training_args = TrainingArguments(
                output_dir = self.config.model_log_dir,
                evaluation_strategy = self.config.evaluation_strategy,
                learning_rate = float(self.config.learning_rate),
                per_device_train_batch_size = self.config.per_device_train_batch_size,
                per_device_eval_batch_size = self.config.per_device_eval_batch_size,
                num_train_epochs = self.config.num_train_epochs,
                weight_decay = self.config.weight_decay,
                save_total_limit = self.config.save_total_limit,
                save_strategy = self.config.save_strategy
            )
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=training_dataset,
                eval_dataset=validation_dataset,
                data_collator=self.data_collector,
                tokenizer=tokenizer,
                compute_metrics=compute_metrics
            )
            trainer.train() # train the model
            log(file_object=log_file, log_message=f"model training done") # logs 

            model.save_pretrained(self.config.doc_tag_model_dir) # save the trained model
            log(file_object=log_file, log_message=f"save the model into {self.config.doc_tag_model_dir}") # logs 

            # store the mapping information:
            label2id = {key:str(val)  for key, val in tag2id.items()}
            id2label = {str(val):key  for key, val in tag2id.items()}
            log(file_object=log_file, log_message=f"get label2id and id2label from tag2id") # log

            config = json.load(open(f"{self.config.doc_tag_model_dir}/config.json"))
            config["id2label"] = id2label
            config["label2id"] = label2id
            json.dump(config, open(f"{self.config.doc_tag_model_dir}/config.json","w"))
            log(file_object=log_file, log_message=f"dump label2id and id2label to {self.config.doc_tag_model_dir}/config.json file") # log 

        except Exception as ex:
            log_file = self.config.log_file # mention log file
            log(file_object=log_file, log_message=f"error will be: {ex}") # logs
            raise ex




if __name__ == "__main__":
    from src.document_tagging.config.configuration import ConfigManager
    config_manager = ConfigManager()
    model_training_config = config_manager.get_model_training_config()

    model = ModelTraining(config=model_training_config)
    model.train_model()
