# BERT Sentiment Analysis on IMDB Dataset

This project contains an implementation of fine-tuning the BERT (Bidirectional Encoder Representations from Transformers) model on the IMDB movie review dataset for sentiment analysis.

## Model

The model used in this project is the BERT model from the `transformers` library. It is a pre-trained model on a large corpus of text data, and fine-tuned on the IMDB dataset for sentiment analysis. 

## Dataset

The IMDB movie review dataset is used in this project. The dataset is split into training and testing sets using a 80:20 ratio. Each review is either labeled as positive or negative, represented as 1 and 0 respectively.

## Results

Due to resource limitations, the model was trained for only one epoch. The result of this training yielded the following:

- Test Loss: 0.3948
- Test Accuracy: 81.13%

Note that the model's performance could likely be improved significantly by training for more epochs. The limited number of epochs in this experiment was due to a lack of computational resources and the time-consuming nature of the training process.

## Dependencies

This project has the following dependencies:

- [PyTorch](https://pytorch.org/)
- [Transformers](https://huggingface.co/transformers/)
- [Pandas](https://pandas.pydata.org/)
- [Scikit-Learn](https://scikit-learn.org/stable/)
- [TQDM](https://tqdm.github.io/)
- [NumPy](https://numpy.org/)
