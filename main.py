
import warnings
warnings.filterwarnings('ignore')
from App import App
CSV_FILE_PATH = 'datasets/wine_quality.csv'


def wine_quality_prediction_custom():
    app = App(CSV_FILE_PATH)
    app.standardize_data()
    app.fit()
    app.test_accuracy_score()
    app.predict()


wine_quality_prediction_custom()


