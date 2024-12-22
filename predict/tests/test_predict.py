import unittest
from unittest.mock import patch, MagicMock
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from predict.run import TextPredictionModel

class TestTextPredictionModel(unittest.TestCase):

    @patch('predict.run.embed')
    @patch('predict.run.models.load_model')
    def setUp(self, mock_load_model, mock_embed):
        self.mock_model = MagicMock()
        mock_load_model.return_value = self.mock_model
        mock_embed.return_value = [[0.1, 0.2, 0.3]]
        self.mock_model.predict.return_value = [[0.1, 0.9, 0.05, 0.05]]
        self.model = TextPredictionModel.from_artefacts('./train/data/artefacts/2024-12-11-12-12-24')
        self.model.labels_to_index = {'label1': 0, 'label2': 1, 'label3': 2, 'label4': 3}
        self.model.labels_index_inv = {0: 'label1', 1: 'label2', 2: 'label3', 3: 'label4'}

    def test_predict(self):
        text_list = ["example text"]
        top_k = 2
        predictions = self.model.predict(text_list, top_k)
        self.assertEqual(predictions, [['label2', 'label1']])

if __name__ == '__main__':
    unittest.main()