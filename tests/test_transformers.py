import array
import unittest

from examples.transformers_example import TransformesExample

class TransformersTestCase(unittest.TestCase):

    def test_sentiment_analysis(self):
        tranformer_example:TransformesExample = TransformesExample()
        result:array = tranformer_example.sentiment_analysis(["I'm so happy", "This sucks", "I don't know how i'm feeling"])

        self.assertIsNotNone(result)
        print('-----')
        print(result)
        self.assertEqual("POSITIVE", result[0]['label'])
        self.assertEqual("NEGATIVE", result[1]['label'])

    def test_classification(self):
        tranformer_example:TransformesExample = TransformesExample()
        result:array = tranformer_example.classification("Colombia president sucks !", ["education", "politics", "business"])
        print('-----')
        print(result)
        self.assertGreaterEqual(0.8, result['scores'][1])


