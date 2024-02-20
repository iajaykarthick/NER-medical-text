# Named Entity Recognition for Medical Text

## Overview
This project develops a Named Entity Recognition (NER) system to identify and classify medical entities such as diseases, symptoms, treatments from unstructured medical text written in natural language. Utilizing state-of-the-art deep learning techniques, this NER system improves the accuracy and efficiency of medical information extraction.

## Abstract
The NER system uses a combination of word embeddings and recurrent neural networks (RNNs) to capture contextual information and neural network architectures, including bidirectional LSTMs, LSTM with Time Distributed Dense Layers, and CNN combinations. The system is further enhanced with fine-tuned language models like Distil BERT for optimal performance.
 
## Introduction
With the vast amount of unstructured medical text data available, there is a critical need for efficient NER systems to extract meaningful information. This project addresses the challenge by developing a robust NER system capable of handling the complexities of medical language.

## Dataset
The project utilizes the MACCROBAT2018 dataset, which includes annotated medical text, allowing for the development of accurate machine learning models for the biomedical field.

## Methodology
Our methodology for developing a Named Entity Recognition (NER) system for medical text involved several strategic steps, each aimed at enhancing the system's ability to accurately identify and classify critical medical entities such as diseases, symptoms, drugs, and procedures.

1. Data Collection and Preprocessing
We sourced a comprehensive dataset of unstructured medical texts. These texts included a wide range of medical records, publications, and reports. The preprocessing phase involved cleaning the data, normalizing terms, and tokenizing the text into analyzable units. This step was critical in ensuring the quality and consistency of the data fed into our models.

2. Implementation of Word Embeddings
Word embeddings were employed to convert words into numerical form so that the deep learning models could process the textual data. We utilized advanced embedding techniques that capture not just the semantic meaning of words but also their contextual relevance in medical texts.

3. Model Architecture Selection
    Our approach included experimenting with various neural network architectures. We explored:

    * Bidirectional Long Short-Term Memory (Bi-LSTM): This architecture   processes data in both directions with two separate hidden layers, which provide additional context and improve the model's learning capacity.

    * LSTM with Time Distributed Dense Layers: This variant involves applying a dense layer to every temporal slice of an input. It's particularly useful for making frame-wise predictions, which is analogous to predicting an entity for each word in a sentence.

    * Combinations of LSTM and Convolutional Neural Networks (CNNs): Integrating LSTM and CNN models allowed us to capture both temporal sequence dependencies and spatial hierarchies in the data.

4. Utilization of Pre-trained Language Models
To further enhance performance, we fine-tuned pre-trained language models such as DistilBERT. These models have been pre-trained on extensive corpuses of text and can be fine-tuned with additional data to perform specific tasks like NER.

5. Model Training and Evaluation
We trained our models using the prepared datasets, continuously monitoring for overfitting and underfitting by validating the performance on a separate validation set. We employed various evaluation metrics, such as Precision, Recall, and the F1-score, to assess the model's entity recognition capabilities.

6. Performance Comparison and Analysis
After training, we conducted a thorough comparison of the different architectures. This analysis helped us identify the most effective model in terms of accuracy, efficiency, and generalization across different medical domains.

7. System Integration and Testing
The final phase involved integrating the best-performing NER model into a system that can process large volumes of unstructured medical text. Rigorous testing was conducted to ensure the system's reliability and accuracy in a real-world medical environment.

By following these methodological steps, we have developed a robust NER system that significantly contributes to the field of medical research and patient care by providing efficient and accurate extraction of medical entities

## Results
The experiments conducted with various models on the test dataset yielded the following results. We evaluated the models based on test loss, test accuracy, test precision, test recall, and test F1 score.

Among the different models tested, Model 5, which is a Bidirectional LSTM with Dropout, Time-Distributed Dense Layer, and Embedding, emerged as the most effective. This model achieved the highest test F1 score of 0.9451, indicating its robust ability to accurately classify text into the appropriate categories. 

Close behind was Model 3, a Pretrained Bidirectional LSTM with Embedding, which also showed strong performance with a test F1 score of 0.9444. 

Overall, all the models demonstrated reasonable effectiveness in the task, with test F1 scores ranging from 0.9381 to 0.9451. This suggests that each model has its merits in handling named entity recognition, though some outperform others in specific metrics.

## Conclusion
The NER system developed in this project shows significant potential in improving medical data extraction processes and aiding medical research.

## Future Work
Potential future improvements include domain-specific pre-trained models and exploring semi-supervised approaches due to the limited availability of labeled medical data.