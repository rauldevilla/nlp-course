import array
from transformers import pipeline

class TransformesExample:

    def sentiment_analysis(self, texts:array) -> array:
        classifier = pipeline("sentiment-analysis")
        return classifier(texts)
    
    def classification(self, text:str, labels:array) -> dict:
        classifier = pipeline("zero-shot-classification")
        return classifier(
            text,
            candidate_labels=labels,
        )