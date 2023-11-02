from torch.utils.data import Dataset

class CustomDataset(Dataset):
    """
        A custom dataset class for getting train_dataset, test_dataset and valid_dataset.

        Args:
            input_ids (list or array-like): A list or array-like object containing the input IDs for each item in the dataset.
            attention_mask (list or array-like): A list or array-like object containing the attention masks for each item in the dataset.
            labels (list or array-like): A list or array-like object containing the labels for each item in the dataset.
    """

    def __init__(self, input_ids, attention_mask, labels):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.labels = labels

    def __len__(self):
        """
            Returns the length of the dataset.

            Returns:
                int: The length of the dataset.
        """
        return len(self.input_ids)

    def __getitem__(self, idx):
        """
            Returns a dictionary containing the input_ids, attention_mask, and labels for the item at the specified index.

            Args:
                idx (int): The index of the item to retrieve.

            Returns:
                dict: A dictionary containing the input_ids, attention_mask, and labels for the item at the specified index.
        """
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "labels": self.labels[idx]
        }
            
        

