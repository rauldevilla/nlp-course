import array
from transformers import pipeline

class TransformesExample:

    def sentiment_analysis(self, texts:array) -> array:
        classifier = pipeline("sentiment-analysis")
        return classifier(texts)
    
    def classification(self, text:str, labels:array) -> dict:
        classifier = pipeline("zero-shot-classification")
        return classifier(text, candidate_labels=labels)
    
    def text_generation(self, prefix:str, max_length:int = 20, num_return_sequences:int = 1, model:str = "distilgpt2") -> array:
        generator = pipeline("text-generation", model = model)
        return generator(prefix, max_length = max_length, num_return_sequences = num_return_sequences)

    def fill_mask(self, masked_text:str, top_k:int = 1) -> array:
        unmasker = pipeline("fill-mask")
        return unmasker(masked_text, top_k = top_k)
