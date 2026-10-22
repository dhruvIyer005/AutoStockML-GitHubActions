import joblib
import numpy as np
from sklearn.linear_model import LinearRegression


def main():
    X = []
    y = []

    for start in range(50, 150):
        prices = [
            start,
            start + 1,
            start + 2,
            start + 1.5,
            start + 3,
        ]
        X.append(prices)
        y.append(start + 4)

    model = LinearRegression()
    model.fit(np.array(X), np.array(y))

    joblib.dump(model, "model.joblib")
    print("Model trained and saved successfully!!!!.")


if __name__ == "__main__":
    main()