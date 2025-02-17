import numpy as np
from typing import List, Tuple
### YOU MANY NOT ADD ANY MORE IMPORTS (you may add more typing imports)

# Attribution dictionary
REFERENCES = {
    "LabelEncoder.transform": "https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html",
    "StandardScaler.transform": "https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html",
    "LabelEncoder.np.searchSorted": "https://numpy.org/doc/2.1/reference/generated/numpy.searchsorted.html",
    "StandardScalar.divide_by_zero": "https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html#:~:text=If%20a%20variance%20is%20zero%2C%20we%20can%E2%80%99t%20achieve%20unit%20variance%2C%20and%20the%20data%20is%20left%20as%2Dis%2C%20giving%20a%20scaling%20factor%20of%201."
}

class MinMaxScaler:
    def __init__(self):
        self.minimum = None
        self.maximum = None
        
    def _check_is_array(self, x: np.ndarray) -> np.ndarray:
        if not isinstance(x, np.ndarray):
            x = np.array(x)
        assert isinstance(x, np.ndarray), "Expected the input to be a list"
        return x
        
    def fit(self, x: np.ndarray) -> None:   
        x = self._check_is_array(x)
        self.minimum = x.min(axis=0)
        self.maximum = x.max(axis=0)
        
    def transform(self, x: np.ndarray) -> np.ndarray:
        x = self._check_is_array(x)
        diff_max_min = self.maximum - self.minimum
        # Corrected the formula by adding parentheses
        return (x - self.minimum) / diff_max_min
    
    def fit_transform(self, x: np.ndarray) -> np.ndarray:
        x = self._check_is_array(x)
        self.fit(x)
        return self.transform(x)
    
class LabelEncoder:
    def __init__(self):
        self.classes_ = None

    def _check_is_array(self, y: np.ndarray) -> np.ndarray:
        if not isinstance(y, np.ndarray):
            y = np.array(y)
        assert isinstance(y, np.ndarray), "Expected the input to be a list"

        # Check for empty array
        if y.size == 0:
            raise ValueError("Input array is empty.")
        
        return y

    def fit(self, y):
        y = self._check_is_array(y)
        if y.ndim != 1:
            raise ValueError("y must be 1-dimensional.")
        if len(y) == 0:
            raise ValueError("y must not be empty.")
        self.classes_ = np.unique(y)
        return self

    def transform(self, y):
        if self.classes_ is None:
            raise ValueError("LabelEncoder has not been fitted yet.")
        y = self._check_is_array(y)
        if y.ndim != 1:
            raise ValueError("y must be 1-dimensional.")
        unseen = np.setdiff1d(y, self.classes_)
        if len(unseen) > 0:
            raise ValueError(f"y contains previously unseen labels: {unseen.tolist()}")

        """
        I have used np.searchsorted(self.classes_, y) which essentially maps each label in y to its index in self.classes_
        Since self.classes_ is sorted (from np.unique in fit), this converts labels to integers like 0, 1, 2, ....
        Example:
        If self.classes_ = [1, 2, 6] (from fit([1, 2, 2, 6])), then:
        transform([1, 1, 2, 6]) returns [0, 0, 1, 2].
        """

        return np.searchsorted(self.classes_, y)

    def fit_transform(self, y):
        return self.fit(y).transform(y)

class StandardScaler:
    # def __init__(self):
        # raise NotImplementedError
    def __init__(self):
        self.mean = None
        self.scale = None  # Standard deviation (sigma)
        
    def _check_is_array(self, x: np.ndarray) -> np.ndarray:
        if not isinstance(x, np.ndarray):
            x = np.array(x)
        assert isinstance(x, np.ndarray), "Expected the input to be a list"

        # Check for empty array
        if x.size == 0:
            raise ValueError("Input array is empty.")

        return x
        
    def fit(self, x: np.ndarray) -> None:
        x = self._check_is_array(x)
        self.mean = np.mean(x, axis=0)
        self.scale = np.std(x, axis=0, ddof=0)  # Population std
        
    def transform(self, x: np.ndarray) -> np.ndarray:
        x = self._check_is_array(x)
        if self.mean is None or self.scale is None:
            raise ValueError("StandardScaler has not been fitted yet.")
        
        # Handle zero standard deviation during transformation
        if isinstance(self.scale, np.ndarray):  # Multiple features
            scale_safe = np.array([1.0 if s == 0 else s for s in self.scale])
        else:  # Single feature
            scale_safe = 1.0 if self.scale == 0 else self.scale

        return (x - self.mean) / scale_safe
    
    def fit_transform(self, x: np.ndarray) -> np.ndarray:
        x = self._check_is_array(x)
        self.fit(x)
        return self.transform(x)