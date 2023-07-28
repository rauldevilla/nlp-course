import array
import unittest

from examples.transformers_example import TransformesExample

class TransformersTestCase(unittest.TestCase):

    tranformer_example:TransformesExample = TransformesExample()

    def setUp(self):
        print('\n-----')

    def test_sentiment_analysis(self):
        result:array = self.tranformer_example.sentiment_analysis(["I'm so happy", "This sucks", "I don't know how i'm feeling"])
        print(result)
        self.assertEqual("POSITIVE", result[0]['label'])
        self.assertEqual("NEGATIVE", result[1]['label'])

    def test_classification(self):
        result:array = self.tranformer_example.classification("Colombia president sucks !", ["education", "politics", "business"])
        print(result)
        self.assertGreaterEqual(0.8, result['scores'][1])

    def test_text_generation(self):
        result:array = self.tranformer_example.text_generation("My country is")
        print(result)
        self.assertEqual(1, len(result))



