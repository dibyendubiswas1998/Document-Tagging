from src.document_tagging.utils.common_utils import log, save_torch_data, save_json_file 
from src.document_tagging.entity.config_entity import DataPreprocessingConfig
import pandas as pd
import string
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from transformers import AutoTokenizer
from ensure import ensure_annotations
import warnings
warnings.filterwarnings("ignore")

nltk.download('punkt')
nltk.download('stopwords')


class DataPreprocessing:
    """
        The `DataPreprocessing` class is responsible for handling the data preprocessing pipeline. It reads data from a CSV file, 
        fills null values using the forward-fill method, removes duplicates, separates the X and Y features, applies text 
        preprocessing, converts tags in the Y feature to numerical IDs, applies word representation techniques, and splits the 
        data into train, validation, and test datasets.

        Example Usage:
            config = DataPreprocessingConfig()
            preprocessing = DataPreprocessing(config)
            preprocessing.process()

        Methods:
            - __init__(self, config: DataPreprocessingConfig) -> None: Initializes the `DataPreprocessing` class with a configuration object.
            - handle_data(self): Processes the data by reading it from a CSV file, filling null values using the forward-fill method, removing duplicates, and returning the processed data.
            - separate_x_y_feature(self, data): Separates the X and Y features from the input data.
            - text_preprocessing(self, data): Preprocesses text data by applying a series of steps such as lowercasing, removing punctuation, removing numbers, removing special characters, word tokenization, and removing stop words.
            - tag2_id(self, Y): Converts the tags in the Y feature of the input data into numerical IDs.
            - word_representation(self, X, Y, tag2id): Applies word representation techniques to the input data.
            - filter_examples(self, input_ids, attention_mask, labels, batch_size): Filters examples based on the size of their labels.
            - split_data(self, input_ids, attention_mask, tag_data_tensor, length_of_tag_data_tensor): Splits data into train, test, and validation datasets.
            - process(self): Handles the data preprocessing pipeline.

        Fields:
            - config: The configuration object for data preprocessing.
    """
    def __init__(self, config: DataPreprocessingConfig) -> None:
        self.config = config
    

   
    def handle_data(self):
        """
            Process the data by reading it from a CSV file, filling null values using the forward-fill method,
            removing duplicates, and returning the processed data.

            Returns:
                pandas.DataFrame: The processed data.

            Raises:
                Exception: If an error occurs during the data processing.

            Example Usage:
                config = DataPreprocessingConfig()
                preprocessing = DataPreprocessing(config)
                data = preprocessing.handle_data()
        """
        try:
            log_file = self.config.log_file # mention log file
            data = pd.read_csv(self.config.data_file_path) # read data
            data.fillna(method='ffill', inplace=True) # fill the null values using forward-fill method
            data.drop_duplicates(inplace=True) # remove duplicates
            log(file_object=log_file, log_message=f"apply the forward-fill method and remove the duplicates") # logs
            return data # return data

        except Exception as ex:
            log_file = self.config.log_file # mention log file
            log(file_object=log_file, log_message=f"error will be: {ex}") # logs
            raise ex
        


    def separate_x_y_feature(self, data):
        """
            Separates the X and Y features from the input data.

            Args:
                data (pd.DataFrame): The input data containing the X and Y features.

            Returns:
                tuple: A tuple containing the X and Y features.

            Example:
                pre = PreProcessing()
                data = pd.DataFrame({'X': [1, 2, 3], 'Y': [4, 5, 6]})
                X, Y = pre.separate_x_y_feature(data)
                print(X)  # Output: [1, 2, 3]
                print(Y)  # Output: [4, 5, 6]
        """
        try:
            log_file = self.config.log_file # mention log file
            X = data[self.config.X_feature_name] # X data
            Y = data[self.config.Y_feature_name] # Y data
            log(file_object=log_file, log_message=f"separate the X and Y data") # logs
            return X, Y # return the X and Y data

        except Exception as ex:
            log_file = self.config.log_file # mention log file
            log(file_object=log_file, log_message=f"error will be: {ex}") # logs
            raise ex
    


    def text_preprocessing(self, data):
        """
            Preprocesses text data by applying a series of steps such as lowercasing, removing punctuation, removing numbers, removing special characters, word tokenization, and removing stop words.

            Args:
                data (list): The input text data to be preprocessed.

            Returns:
                list: The preprocessed text data as a list of sentences.

            Raises:
                Exception: If an error occurs during the text preprocessing steps.

            Example Usage:
                pre = PreProcessing()
                data = ["This is an example sentence.", "Another sentence with numbers 123."]
                preprocessed_data = pre.text_preprocessing(data)
                print(preprocessed_data)
                # Output: ['example sentence', 'another sentence numbers']
        """
        try:
            log_file = self.config.log_file # mention log file
            sentences = data
            preprocessed_sentences = []

            for sentence in sentences:
                # lowering the sentences:
                text = sentence.lower()

                # remove punctuation:
                translator = str.maketrans('', '', string.punctuation)
                text = text.translate(translator)

                # remove the numbers:
                text = re.sub(r'\d+', '', text)

                # Remove special characters
                text = re.sub(r'[^\w\s]', '', text)

                # word tokenization:
                tokens = word_tokenize(text)

                # remove the stop of words:
                words = [word for word in tokens if word not in set(stopwords.words('english'))]

                # get the pre-processed sentences:
                preprocessed_sentence = ' '.join(words)
                preprocessed_sentences.append(preprocessed_sentence)

            log(file_object=log_file, log_message="apply the text-preprocessing steps") # logs 
            return preprocessed_sentences # return data after applying text-preprocessing

        except Exception as ex:
            log_file = self.config.log_file # mention log file
            log(file_object=log_file, log_message=f"error will be: {ex}") # logs
            raise ex
    


    def tag2_id(self, Y):
        """
            Convert the tags in the Y feature of the input data into numerical IDs.

            Args:
                X (pd.Series): The X feature of the input data.
                Y (pd.Series): The Y feature of the input data.

            Returns:
                tuple: A tuple containing:
                    - tag2id (dict): A dictionary where each unique tag is mapped to a unique ID.
                    - num_labels (int): The total number of unique tags.

            Raises:
                Exception: If an error occurs during the conversion process.

            Example:
                pre = PreProcessing()
                data = pd.DataFrame({'X': [1, 2, 3], 'Y': [4, 5, 6]})
                X, Y = pre.separate_x_y_feature(data)
                tag2id, num_labels = pre.tag2_id(X, Y)
                print(tag2id)  # Output: {'4': 0, '5': 1, '6': 2}
                print(num_labels)  # Output: 3
        """
        try:
            log_file = self.config.log_file # mention log file
            y_data_series = pd.Series(Y)
            tags = list(y_data_series.str.split())  # get all the list of tags

            unique_tags = set(tag for tag_list in tags for tag in tag_list)  # get all the unique tags
            tag2id = {tag: idx for idx, tag in enumerate(unique_tags)}  # get the id based on specific tags as a dictionary
            save_json_file(file_path=self.config.tag2id, report=tag2id) # save the tag2id into the file
            log(file_object=log_file, log_message=f"get unique id for each tag and save into {self.config.tag2id}") # logs

            num_labels = len(tag2id)  # get the number of labels
            save_json_file(file_path=self.config.json_file, report={"no_labels": num_labels}) # save the number of labels into json file
            log(file_object=log_file, log_message=f"save the number of labels into {self.config.json_file}") # logs

            return tag2id, num_labels  # return tag2id and num_labels

        except Exception as ex:
            log_file = self.config.log_file # mention log file
            log(file_object=log_file, log_message=f"error will be: {ex}") # logs
            raise ex



    def word_representation(self, X, Y, tag2id):
        """
            Applies word representation techniques to the input data.

            Args:
                X (list): The input text data to be processed.
                Y (list): The tags corresponding to the input text data.
                tag2id (dict): A dictionary mapping tags to numerical IDs.

            Returns:
                input_ids (tensor): The input IDs representing the tokenized text data.
                attention_mask (tensor): The attention masks indicating which tokens to attend to.
                tag_data_tensor (tensor): The numerical IDs representing the tags in the Y feature.
                length_of_tag_data_tensor (int): The length of the tag data tensor.
        """
        try:
            log_file = self.config.log_file # mention log file

            # Apply word_representation on X data:
            log(file_object=log_file, log_message="apply the word representation on X data") # logs 
            tokenizer = AutoTokenizer.from_pretrained(self.config.model_name) # load the tokenizer
            tokenized_titles = tokenizer(X, padding=True, truncation=True, return_tensors='pt') # apply the word representation
        
            input_ids = tokenized_titles['input_ids'] # get the input_ids
            attention_mask = tokenized_titles['attention_mask'] # get the attention_mask
            desired_length = len(input_ids[0]) # get the desired length of input_ids
            log(file_object=log_file, log_message=f"get the input_ids and attention_mask and length of input_ids {desired_length}") # logs about get the input_ids and attention_mask

            tokenizer.save_pretrained(self.config.tokenizer_path) # save the tokenizer
            log(file_object=log_file, log_message=f"save the tokenizer into {self.config.tokenizer_path}") # logs


            # Apply word_representation on Y data:
            log(file_object=log_file, log_message="apply the word representation on Y data") # logs
            y_data_series = pd.Series(Y) 
            tags = list(y_data_series.str.split()) # get all the list of tags

            dd = {self.config.X_feature_name:X, self.config.Y_feature_name:tags} # create data dictionary using the X and tags data
            data = pd.DataFrame(dd) # create data frmae

            data['Tags'] = data['Tags'].apply(lambda tags: [tag2id[tag] for tag in tags if tag in tag2id]) # mapping the tags data with tagid values
            tags_data_padded = [tag_list + [-100] * (desired_length - len(tag_list)) for tag_list in data['Tags']] # pad the lists with -100 to make them all the same length
            tag_data_tensor = torch.tensor(tags_data_padded) # create tensord object
            length_of_tag_data_tensor = len(tag_data_tensor[0]) # get the length of the tag_data_tensor
            log(file_object=log_file, log_message=f"get the length of the tag_data_tensor {length_of_tag_data_tensor}") # logs

            return input_ids, attention_mask, tag_data_tensor, length_of_tag_data_tensor # return the input_ids, attention_mask, tag_data_tensor and length_of_tag_data_tensor.            
    
        except Exception as ex:
            log_file = self.config.log_file # mention log file
            log(file_object=log_file, log_message=f"error will be: {ex}") # logs
            raise ex



    def filter_examples(self, input_ids, attention_mask, labels, batch_size):
        """
            Filters examples based on the size of their labels.

            Args:
                input_ids (list): The input IDs of the examples.
                attention_mask (list): The attention masks of the examples.
                labels (list): The labels of the examples.
                batch_size (int): The desired batch size.

            Returns:
                tuple: A tuple containing the filtered input IDs, attention masks, and labels.

            Raises:
                Exception: If an error occurs during the filtering process.

            Example Usage:
                pre = PreProcessing()
                input_ids = [1, 2, 3, 4, 5]
                attention_mask = [1, 1, 1, 1, 1]
                labels = [[1, 2, 3], [4, 5, 6], [], [7, 8, 9], [10, 11, 12]]
                batch_size = 3
                filtered_input_ids, filtered_attention_mask, filtered_labels = pre.filter_examples(input_ids, attention_mask, labels, batch_size)
                print(filtered_input_ids)  # Output: [1, 2, 4, 5]
                print(filtered_attention_mask)  # Output: [1, 1, 1, 1]
                print(filtered_labels)  # Output: [[1, 2, 3], [7, 8, 9], [10, 11, 12]]
        """
        try:
            # Filter examples with empty labels or labels that don't match batch size
            filtered_input_ids = []
            filtered_attention_mask = []
            filtered_labels = []

            for i in range(len(input_ids)):
                if len(labels[i]) == batch_size:
                    filtered_input_ids.append(input_ids[i])
                    filtered_attention_mask.append(attention_mask[i])
                    filtered_labels.append(labels[i])
                else:
                    print(f"Example {i} has incorrect label size: {len(labels[i])}")

            return filtered_input_ids, filtered_attention_mask, filtered_labels

        except Exception as ex:
            log_file = self.config.log_file # mention log file
            log(file_object=log_file, log_message=f"error will be: {ex}") # logs
            raise ex
    

 
    def split_data(self, input_ids, attention_mask, tag_data_tensor, length_of_tag_data_tensor):
        """
            Split data into train, test, and validation datasets.

            Args:
                input_ids (list): A list of input IDs.
                attention_mask (list): A list of attention masks.
                tag_data_tensor (list): A list of tag data tensors.
                length_of_tag_data_tensor (list): A list of lengths of tag data tensors.

            Returns:
                tuple: A tuple containing the train dataset tensors.
                tuple: A tuple containing the validation dataset tensors.
                tuple: A tuple containing the test dataset tensors.

            Raises:
                Exception: If an error occurs during the process.

            Example Usage:
                pre = PreProcessing()
                input_ids = [[1, 2, 3], [4, 5, 6]]
                attention_mask = [[1, 1, 1], [1, 1, 1]]
                tag_data_tensor = [[0, 1, 0], [1, 0, 1]]
                length_of_tag_data_tensor = [3, 3]

                train, valid, test = pre.create_tensor_and_split_data(input_ids, attention_mask, tag_data_tensor, length_of_tag_data_tensor)
                print(train)  # Output: ([tensor([1, 2, 3]), tensor([4, 5, 6])], [tensor([0, 1, 0]), tensor([1, 0, 1])])
                print(valid)  # Output: ([], [], [])
                print(test)   # Output: ([], [], [])
        """
        try:
            log_file = self.config.log_file # mention log file
            X_input_id_tensor  = [torch.tensor(seq, dtype=torch.long) for seq in input_ids]
            X_attn_mask_tensor = [torch.tensor(seq, dtype=torch.long) for seq in attention_mask]
            Y_tensor = [torch.tensor(seq, dtype=torch.long) for seq in tag_data_tensor]
            log(file_object=log_file, log_message=f"create multiple list of input_id tensor, attention_mask tensor and y tensor object") # logs about the creating multiple list of tensor objects
        
            X_train, X_temp, Y_train, Y_temp, attn_mask_X_train, attn_mask_X_temp = train_test_split(
                X_input_id_tensor,
                Y_tensor,
                X_attn_mask_tensor,
                test_size=self.config.split_ratio,
                random_state=self.config.random_state
            ) # split into train and temp 

            X_valid, X_test, Y_valid, Y_test, attn_mask_X_valid, attn_mask_X_test = train_test_split(
                X_temp,
                Y_temp,
                attn_mask_X_temp,
                test_size=0.5,
                random_state=self.config.random_state
            ) # split into valid and test

            log(file_object=log_file, log_message=f"split the data into train, test and valid.") # logs
            log(file_object=log_file, log_message=f"train-size: {len(X_train)}, test-size: {len(X_test)}, valid-size: {len(X_valid)}") # logs

            # filter examples with empty labels or labels that don't match batch size:
            batch = length_of_tag_data_tensor
            X_train, attn_mask_X_train, Y_train = self.filter_examples(X_train, attn_mask_X_train, Y_train, batch_size=batch)
            X_valid, attn_mask_X_valid, Y_valid = self.filter_examples(X_valid, attn_mask_X_valid, Y_valid, batch_size=batch)
            X_test, attn_mask_X_test, Y_test = self.filter_examples(X_test, attn_mask_X_test, Y_test, batch_size=batch)
            log(file_object=log_file, log_message=f"filter examples with empty labels or labels that don't match batch size") # logs

            return (X_train, attn_mask_X_train, Y_train), (X_valid, attn_mask_X_valid, Y_valid), (X_test, attn_mask_X_test, Y_test) # return the train, valid and test tensor dataset.            
        
        except Exception as ex:
            log_file = self.config.log_file # mention log file
            log(file_object=log_file, log_message=f"error will be: {ex}") # logs
            raise ex


    def process(self):
        """
            Handles the data preprocessing pipeline.

            This method reads the data from a CSV file, fills null values using the forward-fill method, removes duplicates,
            separates the X and Y features, applies text preprocessing on the X and Y features, converts the tags in the Y feature
            to numerical IDs, applies word representation techniques, splits the data into train, validation, and test datasets,
            and saves them.
        """
        try:
            data = self.handle_data() # handle the data
            X, Y = self.separate_x_y_feature(data) # separate the X and Y
            X = self.text_preprocessing(X) # apply text-preprocessing on X
            Y = self.text_preprocessing(Y) # apply text-preprocessing on Y
            tag2id, num_labels = self.tag2_id(Y) # get the unique id for each tag
            input_ids, attention_mask, tag_data_tensor, length_of_tag_data_tensor = self.word_representation(X, Y, tag2id) # apply word to vector
            train, vaild, test = self.split_data(input_ids, attention_mask, tag_data_tensor, length_of_tag_data_tensor) # get the train, test and validation dataset

            save_torch_data(torch_data=train, file_path=self.config.train_torch_file_name) # save train dataset
            save_torch_data(torch_data=vaild, file_path=self.config.valid_torch_file_name) # save train dataset
            save_torch_data(torch_data=test, file_path=self.config.test_torch_file_name) # save train dataset

        except Exception as ex:
            log_file = self.config.log_file # mention log file
            log(file_object=log_file, log_message=f"error will be: {ex}") # logs
            raise ex



if __name__ == "__main__":
    from src.document_tagging.config.configuration import ConfigManager
    config_manager = ConfigManager()
    data_preprocessing_config = config_manager.get_data_preprocessing_config()

    preprocessing = DataPreprocessing(config=data_preprocessing_config)
    # preprocessing.process()

    