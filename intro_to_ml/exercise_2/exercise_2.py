import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import PolynomialFeatures

class LinearRegression:
    def __init__(self, fit_intercept=True):
        self.fit_intercept = fit_intercept
        self.weights = None

    def fit(self, X, y):
        """
        Fits the linear regression model to the training data.

        Args:
            X (numpy.ndarray): Training data features (n_samples, n_features).
            y (numpy.ndarray): Training data target values (n_samples,).

        Returns:
            self: Returns the instance of the LinearRegression class.
        """
        if self.fit_intercept:
            Xa = np.c_[np.ones(X.shape[0]), X]
        else:
            Xa = X

        # Calculate the weights using the normal equation
        self.weights = (np.linalg.inv(Xa.T @ Xa)) @ Xa.T @ y

        return self

    def predict(self, X):
        """
        Predicts target values for new data using the fitted model.

        Args:
            X (numpy.ndarray): New data features (n_samples, n_features).

        Returns:
            numpy.ndarray: Predicted target values (n_samples,).
        """
        if self.weights is None:
            raise ValueError("Model not fitted yet. Call fit() first.")

        if self.fit_intercept:
            Xa = np.c_[np.ones(X.shape[0]), X]
        else:
            Xa = X

            # Make predictions
        y_predicted = Xa @ self.weights
        return y_predicted

    def score(self, X, y):
        """
        Calculates the coefficient of determination (R^2) of the model on the given data.

        Args:
            X (numpy.ndarray): Data features (n_samples, n_features).
            y (numpy.ndarray): True target values (n_samples,).

        Returns:
            float: The R^2 score of the model.
        """
        if self.weights is None:
            raise ValueError("Model not fitted yet. Call fit() first.")

        y_predicted = self.predict(X)

        # Calculate U - the residual sum of squares (RSS)
        U = np.sum((y - y_predicted) ** 2)

        # Calculate V - the total sum of squares (TSS)
        V = np.sum((y - np.mean(y)) ** 2)

        # Calculate R^2
        if V == 0:
            return 1.0 if np.all(y == y_predicted) else 0.0  # Avoid division by zero
        r_squared = round(1 - (U / V), 5)
        return r_squared


def main_3():
    data = pd.read_csv("simple_regression.csv")
    train_data = data.sample(frac=0.8, random_state=42)
    test_data = data.drop(train_data.index)
    model = LinearRegression()
    model.fit(train_data["x"], train_data["y"])
    print("Model coefficients:", model.weights)
    score = model.score(test_data["x"], test_data["y"])
    print("R^2:", score)


def main_4():
    housing = fetch_california_housing(as_frame=True).frame
    train_housing = housing.sample(frac=0.8, random_state=42)
    test_housing = housing.drop(train_housing.index)

    X_train = train_housing.drop(columns=['MedHouseVal']).values
    y_train = train_housing['MedHouseVal'].values
    X_test = test_housing.drop(columns=['MedHouseVal']).values
    y_test = test_housing['MedHouseVal'].values

    model = LinearRegression()
    model.fit(X_train, y_train)
    print("Model coefficients:", model.weights)
    score = model.score(X_test, y_test)
    print("R^2:", score)


def main_5():
    # predict students salary
    predict_students_salary()

    # explore polynomial scores
    dgree_1_score, dgree_2_score, dgree_3_score, dgree_4_score = polynomial_degrees_exploration()

    # plot polynomial scores
    plot_scores(dgree_1_score, dgree_2_score, dgree_3_score)



def student_prediction(poly, model, student):
    print("\nFor student with the following information:")
    print(student)

    # Drop target variable to get input features
    student_info = student.drop("y")
    student_salary = student["y"]

    print("\nThe predicted salary is:")
    student_info_df = pd.DataFrame([student_info], columns=student_info.index)
    student_info_poly = poly.transform(student_info_df) # reshape to 2D array
    print(model.predict(student_info_poly)[0])

    print("The real salary is:")
    print(student_salary)


def predict_students_salary():
    data = pd.read_csv("Students_on_Mars.csv")

    # split the data
    train_data = data.sample(frac=0.8, random_state=42)
    test_data = data.drop(train_data.index)

    X_train = train_data.drop(columns=["y"])
    y_train = train_data["y"]

    poly = PolynomialFeatures(2)
    X_poly_train = poly.fit_transform(X_train)

    model = LinearRegression(fit_intercept=False)
    model.fit(X_poly_train, y_train)

    # predict for first 5 students in test set
    print("Predictions for the first 5 students in the test set:")
    for _, student in test_data.head(5).iterrows():
        student_prediction(poly, model, student)

def get_polynomial_score(polynomial_degree=2):
    data = pd.read_csv("Students_on_Mars.csv")
    # split the data into training and testing sets
    train_data = data.sample(frac=0.8, random_state=42)
    test_data = data.drop(train_data.index)

    # fit
    X_train = train_data.drop(columns=["y"])
    y_train = train_data["y"]

    poly = PolynomialFeatures(polynomial_degree)
    X_poly_train = poly.fit_transform(X_train)

    model = LinearRegression(fit_intercept=False)
    model.fit(X_poly_train, y_train)

    # convert test data to polynomial
    X_test = test_data.drop(columns=["y"])
    y_test = test_data["y"]
    X_test_poly = poly.transform(X_test)

    # get score
    score = model.score(X_test_poly, y_test)  # Use transformed test data
    print("R^2:", score)
    return score


def polynomial_degrees_exploration():
    print("\nPolynomial degrees exploration:")
    print("Polynomial degree 1 - linear baseline:")
    dgree_1_score = get_polynomial_score(polynomial_degree=1)
    print("\nPolynomial degree 2 - significantly increase, almost perfect score:")
    dgree_2_score = get_polynomial_score(polynomial_degree=2)
    print("\nPolynomial degree 3 - slightly better, not significantly:")
    dgree_3_score = get_polynomial_score(polynomial_degree=3)
    print("\nPolynomial degree 4 - overfitting:")
    dgree_4_score = get_polynomial_score(polynomial_degree=4)

    print(
        f"\nThe best polynomial degree in my opinion is 2, which gives an almost perfect score of {dgree_2_score}.\nThe 3th degree score is slightly lower, and the 4th degree score represents overfitting.")

    return dgree_1_score, dgree_2_score, dgree_3_score, dgree_4_score

def plot_scores(dgree_1_score, dgree_2_score, dgree_3_score):
    # plot the scores
    plt.style.use('ggplot')
    plt.figure(figsize=(10, 6))
    plt.plot([1, 2, 3], [dgree_1_score, dgree_2_score, dgree_3_score], marker='o')
    plt.title('Polynomial Regression Scores')
    plt.xlabel('Polynomial Degree')
    plt.ylabel('R^2 Score')
    plt.xticks([1, 2, 3])
    plt.grid()
    plt.show()




if __name__ == '__main__':
    # Question 3
    print("Question 3 - Simple Linear Regression:")
    main_3()

    # Question 4
    print("\nQuestion 4 - prediction of house prices:")
    main_4()

    # Question 5
    print("\nQuestion 5 - Polynomial Regression:")
    main_5()

