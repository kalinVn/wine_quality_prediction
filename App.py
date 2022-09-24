import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier


class App:

    def __init__(self, csv_path):
        self.standard_data = None
        self.x = None
        self.y = None
        self.x_train = None
        self.x_test = None
        self.y_train = None
        self.y_test = None

        self.csv_path = csv_path
        self.dataset = pd.read_csv(csv_path)

        self.model = RandomForestClassifier()

    def standardize_data(self):
        self.x = self.dataset.drop('quality', axis=1)

        self.y = self.dataset['quality'].apply(lambda y_val: 1 if y_val > 7 else 0)
        # y_debug = wine_dataset['quality'].apply(lambda y_val: 1 if y_val > 7 else 0)

    def fit(self):
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.x, self.y, test_size=0.1,
                                                                                random_state=2)

        self.model.fit(self.x_train, self.y_train)

    def predict(self):

        prediction = self.model.predict(self.get_input_data())

        if prediction[0] == 1:
            print('Good quality wine.')
        else:
            print('Bad quality wine.')

    def test_accuracy_score(self):
        x_test_prediction = self.model.predict(self.x_test)
        test_data_accuracy = accuracy_score(x_test_prediction, self.y_test)
        print('Accuracy on test data : ', test_data_accuracy)

        # accuracy on training data
        x_train_prediction = self.model.predict(self.x_train)
        training_data_accuracy = accuracy_score(x_train_prediction, self.y_train)
        print('Accuracy on training data : ', training_data_accuracy)

    def get_input_data(self):
        # Making predictive system
        input_data = (7.4, 0.7, 0.0, 1.9, 0.076, 11.0, 34.0, 0.9978, 3.51, 0.56, 9.4)

        # changing the input data to numpy array
        input_data_numpy_arr = np.asarray(input_data)

        # reshape the np array
        input_data_reshaped = input_data_numpy_arr.reshape(1, -1)
        # print(input_data_reshaped)

        return input_data_reshaped

    def predict(self):

        prediction = self.model.predict(self.get_input_data())

        if prediction[0] == 1:
            print('Good quality wine.')
        else:
            print('Bad quality wine.')



