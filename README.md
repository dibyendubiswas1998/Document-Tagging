# Document Tagging

## Problem Statement:
Document-Tagging system that leverages advanced natural language processing (NLP) techniques and transformer-based pre-trained models for efficient document categorization. This system aims to automatically generate the relevant tags or categories based on large volumes of unstructured textual data. Users can upload documents, which undergo preprocessing and are then classified into multiple relevant categories using NLP models. [Download Problem Statement PDF](./documents/problem%20statement.pdf)

Here I build CI-CD pipeline for generating relevant tags based on given textual data.<br>
[DagsHub](https://dagshub.com/dibyendubiswas1998/Document-Tagging.mlflow)
<br><br>

## Project Documentation:
[High Level Design Dcocument (HLD)](./documents/High%20Level%20Design%20(HLD).pdf)

[Architecture Document](./documents/Architecture.pdf)

[Wireframe Document](./documents/Wireframe.pdf)

[Low Level Design Document (LLD)](./documents/Low%20Level%20Design%20(LLD).pdf)

[Detailed Project Report (DPR)](./documents/Detailed%20Project%20Report%20(DPR).docx.pdf)
<br><br>


## Project Workflow:
* **Step-01:** Load the raw or custom data from **AWS S3 Bucket**, provided by user. And save the data into particular directory.

* **Step-02:** Preprocessed the raw data, like handle the missing values, duplicate values, text-preprocessing, vectorization, separate the X and Y, create tensor dataset and split them into train, test and validation sets.

* **Step-03:** Create the model (default: bert-base-uncased), and train the model. After that save the pre-trained model & tokenizer in a particular directory.

* **Step-04:** Evaluate the model baed on test datasets and save the inflormation on [DagsHub](https://dagshub.com/dibyendubiswas1998/Document-Tagging.mlflow) using mlflow. 

* **Step-05:** Create a Web Application for generating the tags and host the entier application on AWS.
<br><br>



## Tech Stack:
![Tech Stack](./documents/tech%20stack.png)
<br><br>


## How to Run the Application:
```bash
    # For Windows OS:
    docker pull dibyendubiswas1998/document_tagging
    docker run -p 8080:8080 dibyendubiswas1998/document_tagging

    # For Ubuntu OS:
    sudo docker pull dibyendubiswas1998/document_tagging
    sudo docker run -p 8080:8080 dibyendubiswas1998/document_tagging

```
<br><br>


## Web Interface:
![Web Interface](./documents/web%20interface.png)
