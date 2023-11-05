from src.document_tagging.utils.common_utils import log
from transformers import AutoModelForTokenClassification, AutoTokenizer
from transformers import pipeline



class PredictionPipeline:
    def __init__(self):
        """
            Initializes a new instance of the PredictionPipeline class.
        
            Args:
                None
        
            Returns:
                None
        
            Summary:
            The __init__ method initializes the object by setting the paths for the model file and tokenizer file.
        
            Example Usage:
            pipeline_obj = PredictionPipeline()
        
            Code Analysis:
                - The __init__ method is called when a new instance of the PredictionPipeline class is created.
                - It sets the model_file_path attribute to the string "artifacts/model/doc_tag_model".
                - It sets the tokenizer_file_path attribute to the string "artifacts/tokenizer".
        """
        self.model_file_path = "artifacts/model/doc_tag_model"
        self.tokenizer_file_path = "artifacts/tokenizer"
    
    def prediction(self, data):
        """
            Predicts the tags/entities present in the given input data using a pre-trained model and tokenizer.

            Args:
                data (str): The input data for which the tags/entities need to be predicted.

            Returns:
                set: The predicted tags/entities in the input data.

            Raises:
                Exception: If an error occurs during the prediction process.

            Example Usage:
                pipeline_obj = PredictionPipeline()  # Initialize the PredictionPipeline object
                data = "This is a sample input."  # Input data
                tags = pipeline_obj.prediction(data)  # Predict the tags/entities in the input data
                print(tags)  # Print the predicted tags/entities
        """
        try:
            model = AutoModelForTokenClassification.from_pretrained(self.model_file_path) # load the model
            tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_file_path) # load the tokenizer
            nlp = pipeline(task='ner', model=model, tokenizer=tokenizer) # create a pipeline
            results = nlp(data) # get the results
            filter_data = ['references', 'reff', 'enc', 'reference']
            tags = [dct['entity'] for dct in results if dct['entity'] not in filter_data]
            return set(tags) # return unique tags

        except Exception as ex:
            raise ex
        


if __name__ == "__main__":
    pp = PredictionPipeline()
    rs = pp.prediction("Machine Learning india weipedia")
    print(rs)
