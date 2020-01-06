import unittest
from unittest.mock import MagicMock, patch

import tensorflow as tf
import pandas as pd
import numpy as np

from deepkt import data_util


class TestDataUtil(unittest.TestCase):
    @staticmethod
    def _gen_encoded_data(nb_students, nb_answers, nb_skills):
        preds = np.random.rand(nb_students, nb_answers, nb_skills)
        labels = np.random.randint(2, size=(nb_students, nb_answers))
        inputs = np.zeros(shape=(nb_students, nb_answers, nb_skills+1),
                          dtype=np.float)
        for s in range(nb_students):
            for a in range(nb_answers):
                skill_id = np.random.randint(nb_skills)
                inputs[s, a, skill_id] = 1.
                inputs[s, a, -1] = labels[s, a]
        return inputs, preds, labels

    def test_get_target_must_keepdims(self):
        inputs, preds, labels = self._gen_encoded_data(3, 4, 2)
        y_true, y_pred = data_util.get_target(inputs, preds)

        self.assertEqual(y_true.shape, (3, 4, 1))
        self.assertEqual(y_pred.shape, (3, 4, 1))

    def test_get_target_must_return_values_for_skill_only(self):
        inputs, preds, labels = self._gen_encoded_data(3, 4, 2)

        y_true, y_pred = data_util.get_target(inputs, preds)
        y_true, y_pred = np.squeeze(y_true.numpy()), np.squeeze(y_pred.numpy())

        self.assertEqual(y_true.tolist(), labels.tolist())
        for s in range(3):
            for q in range(4):
                skill = next(p for p, vl in enumerate(
                    inputs[s, q, 0:-1]) if vl == 1)
                self.assertEqual(y_pred[s, q], preds[s, q, skill])

    @patch('deepkt.data_util.pd.read_csv')
    def test_load_dataset_should_return_nb_features(self, mock_pd):
        mock_pd.return_value = pd.DataFrame(
            {'user_id':  pd.Series([1., 1., 1., 2., 2., 3.]),
             'skill_id': pd.Series([1., 2., 3., 4., 3., 1.]),
             'correct':  pd.Series([1., 0., 1., 0., 0., 1.])
             })

        _, _, nb_feat, _ = data_util.load_dataset("", 1, False)

        self.assertEqual(nb_feat, 7)

    @patch('deepkt.data_util.pd.read_csv')
    def test_load_dataset_should_return_nb_skills(self, mock_pd):
        mock_pd.return_value = pd.DataFrame(
            {'user_id':  pd.Series([1., 1., 1., 2., 2., 3.]),
             'skill_id': pd.Series([1., 2., 3., 4., 3., 1.]),
             'correct':  pd.Series([1., 0., 1., 0., 0., 1.])
             })

        _, _, _, nb_skills = data_util.load_dataset("", 1, False)
        self.assertEqual(nb_skills, 4)

    @patch('deepkt.data_util.pd.read_csv')
    def test_load_dataset_should_return_data_size(self, mock_pd):
        mock_pd.return_value = pd.DataFrame(
            {'user_id':  pd.Series([1., 1., 1., 2., 2., 3.]),
             'skill_id': pd.Series([1., 2., 3., 4., 3., 1.]),
             'correct':  pd.Series([1., 0., 1., 0., 0., 1.])
             })

        _, length, _, _ = data_util.load_dataset("", 1, False)
        self.assertEqual(length, 3)

    @patch('deepkt.data_util.pd.read_csv')
    def test_load_dataset_should_return_nb_batch_when_data_is_combined(self, mock_pd):
        mock_pd.return_value = pd.DataFrame(
            {'user_id':  pd.Series([1., 1., 1., 2., 2., 3., 4.]),
             'skill_id': pd.Series([1., 2., 3., 4., 3., 1., 2.]),
             'correct':  pd.Series([1., 0., 1., 0., 0., 1., 1.])
             })

        _, length, _, _ = data_util.load_dataset("", 2, False)
        self.assertEqual(length, 2)

    @patch('deepkt.data_util.pd.read_csv')
    def test_load_dataset_should_drop_last_batch_with_incorrect_size(self, mock_pd):
        mock_pd.return_value = pd.DataFrame(
            {'user_id':  pd.Series([1., 1., 1., 2., 2., 3.]),
             'skill_id': pd.Series([1., 2., 3., 4., 3., 1.]),
             'correct':  pd.Series([1., 0., 1., 0., 0., 1.])
             })

        _, length, _, _ = data_util.load_dataset("", 2, False)
        self.assertEqual(length, 1)

    @patch('deepkt.data_util.pd.read_csv')
    def test_load_dataset_should_encode_skill_in_label(self, mock_pd):
        mock_pd.return_value = pd.DataFrame(
            {'user_id':  pd.Series([1., 1., 1., 2., 2., 3.]),
             'skill_id': pd.Series([1., 2., 3., 4., 3., 1.]),
             'correct':  pd.Series([1., 0., 1., 0., 0., 1.])
             })

        dataset, _, _, _ = data_util.load_dataset("", 1, False)
        it = iter(dataset)

        # Student 1
        inputs, targets = next(it)
        self.assertEqual((1, 3, 5), targets.shape)
        self.assertEqual(
            [[
                [1, 0, 0, 0, 1],
                [0, 1, 0, 0, 0],
                [0, 0, 1, 0, 1]
            ]],
            targets.numpy().tolist())

        # Student 2
        inputs, targets = next(it)
        self.assertEqual((1, 2, 5), targets.shape)
        self.assertEqual(
            [[
                [0, 0, 0, 1, 0],
                [0, 0, 1, 0, 0]
            ]],
            targets.numpy().tolist())

        # Student 3
        inputs, targets = next(it)
        self.assertEqual((1, 1, 5), targets.shape)
        self.assertEqual(
            [[
                [1, 0, 0, 0, 1]
            ]],
            targets.numpy().tolist())

    @patch('deepkt.data_util.pd.read_csv')
    def test_load_dataset_should_pad_batch(self, mock_pd):
        mock_pd.return_value = pd.DataFrame(
            {'user_id':  pd.Series([1., 1., 1., 2., 2., 3.]),
             'skill_id': pd.Series([1., 2., 3., 4., 3., 1.]),
             'correct':  pd.Series([1., 0., 1., 0., 0., 1.])
             })

        dataset, _, _, _ = data_util.load_dataset("", 3, False)
        it = iter(dataset)

        inputs, targets = next(it)
        self.assertEqual((3, 3, 7), inputs.shape)
        self.assertEqual((3, 3, 5), targets.shape)

        # Student 2
        std_2 = targets[1]
        self.assertEqual(
            [
                [0, 0, 0, 1, 0],
                [0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0]
            ],
            std_2.numpy().tolist())

        # Student 3
        std_3 = targets[2]
        self.assertEqual(
            [
                [1, 0, 0, 0, 1],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0]
            ],
            std_3.numpy().tolist())

    def test_split_dataset_with_validation_set(self):
        dataset = tf.data.Dataset.from_tensor_slices((
            [1, 2, 3, 4],
            [5, 6, 7, 8]
        ))

        train, test, val = data_util.split_dataset(dataset, 4, .25, .25)

        self.assertEqual(1, len(list(test)))
        self.assertEqual(1, len(list(val)))
        self.assertEqual(2, len(list(train)))

    def test_split_dataset_without_validation_set(self):
        dataset = tf.data.Dataset.from_tensor_slices((
            [1, 2, 3, 4],
            [5, 6, 7, 8]
        ))

        train, test, val = data_util.split_dataset(dataset, 4, .25)

        self.assertEqual(1, len(list(test)))
        self.assertEqual(3, len(list(train)))
        self.assertIsNone(val)

    def test_split_dataset_should_raise_when_test_frac_bigger_than_allowed(self):
        dataset = tf.data.Dataset.from_tensor_slices((
            [1, 2, 3, 4],
            [5, 6, 7, 8]
        ))

        self.assertRaises(
            ValueError, data_util.split_dataset, dataset, 4, 1.01)

    def test_split_dataset_should_raise_when_test_frac_slower_than_allowed(self):
        dataset = tf.data.Dataset.from_tensor_slices((
            [1, 2, 3, 4],
            [5, 6, 7, 8]
        ))

        self.assertRaises(
            ValueError, data_util.split_dataset, dataset, 4, -0.01)

    def test_split_dataset_should_raise_when_val_frac_bigger_than_allowed(self):
        dataset = tf.data.Dataset.from_tensor_slices((
            [1, 2, 3, 4],
            [5, 6, 7, 8]
        ))

        self.assertRaises(ValueError, data_util.split_dataset,
                          dataset, 4, 0.2, 1.01)

    def test_split_dataset_should_raise_when_val_frac_slower_than_allowed(self):
        dataset = tf.data.Dataset.from_tensor_slices((
            [1, 2, 3, 4],
            [5, 6, 7, 8]
        ))

        self.assertRaises(ValueError, data_util.split_dataset,
                          dataset, 4, 0.2, -0.01)

    def test_split_dataset_should_raise_when_dataset_is_left_with_no_elements(self):
        dataset = tf.data.Dataset.from_tensor_slices((
            [1, 2, 3, 4],
            [5, 6, 7, 8]
        ))

        self.assertRaises(ValueError, data_util.split_dataset,
                          dataset, 4, 0.9)


if __name__ == '__main__':
    unittest.main()
