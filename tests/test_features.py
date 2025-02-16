from aim5005.features import MinMaxScaler, StandardScaler, LabelEncoder
import numpy as np
import unittest
from unittest.case import TestCase

### TO NOT MODIFY EXISTING TESTS

class TestFeatures(TestCase):
    def test_initialize_min_max_scaler(self):
        scaler = MinMaxScaler()
        assert isinstance(scaler, MinMaxScaler), "scaler is not a MinMaxScaler object"
        
        
    def test_min_max_fit(self):
        scaler = MinMaxScaler()
        data = [[-1, 2], [-0.5, 6], [0, 10], [1, 18]]
        scaler.fit(data)
        assert (scaler.maximum == np.array([1., 18.])).all(), "scaler fit does not return maximum values [1., 18.] "
        assert (scaler.minimum == np.array([-1., 2.])).all(), "scaler fit does not return maximum values [-1., 2.] " 
        
        
    def test_min_max_scaler(self):
        data = [[-1, 2], [-0.5, 6], [0, 10], [1, 18]]
        expected = np.array([[0., 0.], [0.25, 0.25], [0.5, 0.5], [1., 1.]])
        scaler = MinMaxScaler()
        scaler.fit(data)
        result = scaler.transform(data)
        assert (result == expected).all(), "Scaler transform does not return expected values. All Values should be between 0 and 1. Got: {}".format(result.reshape(1,-1))
        
    def test_min_max_scaler_single_value(self):
        data = [[-1, 2], [-0.5, 6], [0, 10], [1, 18]]
        expected = np.array([[1.5, 0.]])
        scaler = MinMaxScaler()
        scaler.fit(data)
        result = scaler.transform([[2., 2.]]) 
        assert (result == expected).all(), "Scaler transform does not return expected values. Expect [[1.5 0. ]]. Got: {}".format(result)
        
    def test_standard_scaler_init(self):
        scaler = StandardScaler()
        assert isinstance(scaler, StandardScaler), "scaler is not a StandardScaler object"
        
    def test_standard_scaler_get_mean(self):
        scaler = StandardScaler()
        data = [[0, 0], [0, 0], [1, 1], [1, 1]]
        expected = np.array([0.5, 0.5])
        scaler.fit(data)
        assert (scaler.mean == expected).all(), "scaler fit does not return expected mean {}. Got {}".format(expected, scaler.mean)
        
    def test_standard_scaler_transform(self):
        scaler = StandardScaler()
        data = [[0, 0], [0, 0], [1, 1], [1, 1]]
        expected = np.array([[-1., -1.], [-1., -1.], [1., 1.], [1., 1.]])
        scaler.fit(data)
        result = scaler.transform(data)
        assert (result == expected).all(), "Scaler transform does not return expected values. Expect {}. Got: {}".format(expected.reshape(1,-1), result.reshape(1,-1))
        
    def test_standard_scaler_single_value(self):
        data = [[0, 0], [0, 0], [1, 1], [1, 1]]
        expected = np.array([[3., 3.]])
        scaler = StandardScaler()
        scaler.fit(data)
        result = scaler.transform([[2., 2.]]) 
        assert (result == expected).all(), "Scaler transform does not return expected values. Expect {}. Got: {}".format(expected.reshape(1,-1), result.reshape(1,-1))

    # TODO: Add a test of your own below this line

    # Test cases for LabelEncoder
    def test_label_encoder_numeric_labels(self):
        le = LabelEncoder()
        y = [1, 2, 2, 6]
        le.fit(y)
        assert np.array_equal(le.classes_, np.array([1, 2, 6])), "Test case 1 failed: classes_"
        transformed = le.transform([1, 1, 2, 6])
        assert np.array_equal(transformed, np.array([0, 0, 1, 2])), "Test case 1 failed: transform"

    def test_label_encoder_string_labels(self):
        le = LabelEncoder()
        y = ["paris", "paris", "tokyo", "amsterdam"]
        le.fit(y)
        expected_classes = np.array(["amsterdam", "paris", "tokyo"])
        assert np.array_equal(le.classes_, expected_classes), "Test case 2 failed: classes_"
        transformed = le.transform(["tokyo", "tokyo", "paris"])
        assert np.array_equal(transformed, np.array([2, 2, 1])), "Test case 2 failed: transform"

    def test_label_encoder_fit_transform(self):
        le = LabelEncoder()
        y = [3, 1, 3, 2]
        transformed = le.fit_transform(y)
        assert np.array_equal(le.classes_, np.array([1, 2, 3])), "Test case 3 failed: classes_"
        assert np.array_equal(transformed, np.array([2, 0, 2, 1])), "Test case 3 failed: fit_transform"

    def test_label_encoder_unseen_label_error(self):
        le = LabelEncoder()
        le.fit([1, 2, 3])
        with self.assertRaises(ValueError) as context:
            le.transform([4])
        self.assertTrue("unseen labels" in str(context.exception)), "Test case 4 failed: Error message incorrect"

    def test_label_encoder_empty_y_error(self):
        le = LabelEncoder()
        with self.assertRaises(ValueError) as context:
            le.fit([])
        self.assertTrue("empty" in str(context.exception)), "Test case 5 failed: Error message incorrect"

    def test_label_encoder_2d_input_error(self):
        le = LabelEncoder()
        with self.assertRaises(ValueError) as context:
            le.fit([[1, 2], [3, 4]])
        self.assertTrue("1-dimensional" in str(context.exception)), "Test case 6 failed: Error message incorrect"

if __name__ == '__main__':
    unittest.main()