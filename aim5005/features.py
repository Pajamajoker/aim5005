import numpy as np
from typing import List, Tuple
### YOU MANY NOT ADD ANY MORE IMPORTS (you may add more typing imports)

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

    def _check_is_array(self, y):
        if not isinstance(y, np.ndarray):
            y = np.array(y)
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
        return np.searchsorted(self.classes_, y)

    def fit_transform(self, y):
        return self.fit(y).transform(y)

class StandardScaler:
    def __init__(self):
        raise NotImplementedError