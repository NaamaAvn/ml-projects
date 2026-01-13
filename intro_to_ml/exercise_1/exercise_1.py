import pandas as pd
import random

class Perceptron:
    """Perceptron classifier"""

    def __init__(self, X: pd.DataFrame):
        """Initializes the Perceptron with zero weights and bias."""
        self.weights = pd.Series([0] * X.shape[1], index=X.columns)
        self.b = 0

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Trains the Perceptron model."""
        for i in range(X.shape[0]):
            y_pred = self.predict(X.iloc[i])
            if y[i] != y_pred:
                self.weights += (y[i] * X.iloc[i])
                self.b += y[i]

    def predict(self, sample: pd.Series) -> int:
        """Predicts the class label for a single sample."""
        pred = (self.weights * sample).sum() + self.b
        return 1 if pred > 0 else -1

    def score(self, X: pd.DataFrame, y: pd.Series) -> float:
        """Calculates the accuracy of the model."""
        correct = 0
        for i in range(X.shape[0]):
            y_pred = self.predict(X.iloc[i])
            if y_pred == y[i]:
                correct += 1
        return correct / X.shape[0]

    def error(self, X: pd.DataFrame, y: pd.Series) -> float:
        """Calculates the error rate of the model."""
        return 1 - self.score(X, y)

def encode_labels(y: pd.Series) -> pd.Series:
  """Encodes labels to -1 and 1."""
  return y.map({0: -1, 1: 1})


def split_set(data_set: pd.DataFrame, test_size: float=0.2, random_state: int=42):
    """
    Splits a set of data into a training set and a testing set.
    """
    if random_state is not None:
        random.seed(random_state)
    shuffled_df = data_set.sample(frac=1, random_state=random_state).reset_index(drop=True)
    test_size_index = int(len(shuffled_df) * test_size)
    test_df = shuffled_df[:test_size_index]
    train_df = shuffled_df[test_size_index:]

    return train_df, test_df


def compute_test_score(perceptron: Perceptron, test_set: pd.DataFrame):
    """Computes the test score of a Perceptron model."""
    print(f"\nComputing Test Score on Test Set containing {test_set.shape[0]} samples")
    X_test = test_set.iloc[:, :-1]
    y_test = test_set.iloc[:, -1].reset_index(drop=True)
    y_test_encoded = encode_labels(y_test)
    test_score = perceptron.score(X_test, y_test_encoded)
    test_error = perceptron.error(X_test, y_test_encoded)

    print(f"Test Score: {test_score:.3f}")
    print(f"Test Error: {test_error:.3f}")


def train_perceptron_until_error(perceptron, X_train, y_train_encoded, target_error=0.07):
    """Trains a Perceptron until the error is below a target value"""
    print(f"\nTraining a Perceptron until the error is below a target value {target_error}")
    best_score = 0.0
    best_weights = None
    best_b = None
    epoch = 0
    train_error = 1.0
    while train_error > target_error:
        perceptron.fit(X_train, y_train_encoded)
        epoch_score = perceptron.score(X_train, y_train_encoded)
        train_error = perceptron.error(X_train, y_train_encoded)

        if epoch_score > best_score:
            best_score = epoch_score
            print(f"Best Epoch so far: {epoch}")
            print(f"Score: {epoch_score:.3f}")
            best_weights = perceptron.weights
            best_b = perceptron.b
        epoch += 1

    perceptron.weights = best_weights
    perceptron.b = best_b

    print(f"\nTraining finished after {epoch} epochs.")
    print(f"Train Error: {perceptron.error(X_train, y_train_encoded):.3f}")

    return perceptron


if __name__ == "__main__":
    data = pd.read_csv("./Breast Cancer.csv")
    train_set, test_set = split_set(data)

    # Split the data
    X_train = train_set.iloc[:, :-1]
    y_train = train_set.iloc[:, -1].reset_index(drop=True)
    y_train_encoded = encode_labels(y_train)

    # Initialization
    perceptron = Perceptron(X_train)
    best_score = 0.0
    best_weights = None
    best_b = None

    # Train
    print(f"Training Perceptron on Train Set containing {train_set.shape[0]} samples")
    epoch = 0
    train_error = 1.0
    while train_error > 0 and epoch < 100:
        perceptron.fit(X_train, y_train_encoded)
        epoch_score = perceptron.score(X_train, y_train_encoded)
        train_error = perceptron.error(X_train, y_train_encoded)
        if epoch_score > best_score:
            best_score = epoch_score
            print(f"Best Epoch so far: {epoch}")
            print(f"Score: {epoch_score:.3f}")
            best_weights = perceptron.weights
            best_b = perceptron.b
        epoch +=1

    perceptron.weights = best_weights
    perceptron.b = best_b

    print("\nBest Weights:")
    print(best_weights)
    print("Best Bias:")
    print(best_b)

    print(f"\nTraining finished after {epoch} epochs.")
    print(f"Train Error: {perceptron.error(X_train, y_train_encoded):.3f}")

    # Test
    compute_test_score(perceptron, test_set)



    ##########
    # B Part
    ##########
    perceptron = Perceptron(X_train)
    perceptron = train_perceptron_until_error(perceptron, X_train, y_train_encoded, target_error=0.07)

    print("The best linear separator found:")
    print("\nBest Weights:")
    print(perceptron.weights)
    print("Best Bias:")
    print(perceptron.b)

    compute_test_score(perceptron, test_set)