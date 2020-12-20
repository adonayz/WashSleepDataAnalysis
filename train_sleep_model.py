import pandas as pd
import pickle
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import sys
import prepare_data
from impyute.imputation.cs import mice

sys.setrecursionlimit(100000)  # Increase the recursion limit of the OS

training_dataset_output_path = "data_files/training_data/"


def plot_dataframe(plot_df):
    df = plot_df.copy()
    df = df.reset_index()
    print(df)
    df.plot(x='index', y='sleep_or_wake', kind='line')
    plt.show()


def save_model(clf):
    Pkl_Filename = "frozen_variables/Pickle_" + get_classifier_name(clf) + ".pkl"

    with open(Pkl_Filename, 'wb') as file:
        pickle.dump(clf, file)


def train_data(clf, X, y):
    print("Splitting into train and test data")
    train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.2)

    print("Fitting " + get_classifier_name(clf) + "...")
    clf.fit(train_X, train_y)

    print("Generating prediction...")
    clfPred = clf.predict(test_X)

    print('\nAccuracy:', accuracy_score(test_y, clfPred))
    print('F1 score:', f1_score(test_y, clfPred))
    print('Recall:', recall_score(test_y, clfPred))
    print('Precision:', precision_score(test_y, clfPred))
    print('\n clasification report:\n', classification_report(test_y, clfPred))
    print('\n confussion matrix:\n', confusion_matrix(test_y, clfPred))

    # save_model(clf)


def load_model(clf, X, y):
    print("Loading saved " + get_classifier_name(clf) + " model...")
    Pkl_Filename = "frozen_variables/Pickle_" + get_classifier_name(clf) + ".pkl"
    # Load the Model back from file
    with open(Pkl_Filename, 'rb') as file:
        Pickled_Ada_Model = pickle.load(file)

    score = Pickled_Ada_Model.score(X, y)
    # Print the Score
    print("Test score: {0:.2f} %".format(100 * score))

    # Predict the Labels using the reloaded Model
    Ypredict = Pickled_Ada_Model.predict(X)


def generate_training_data():
    print("Retrieving dataset...")
    dataset = prepare_data.get_sleep_model_training_data()

    print("Cleaning and preparing dataframe")

    dataset['sleep_or_wake'] = (dataset['sleep_or_wake'] == 'S').astype(int)

    # handle missing values
    # from sklearn.impute import SimpleImputer
    # imp_mean = SimpleImputer( strategy='mean') #for median imputation replace 'mean' with 'median'
    # imp_mean.fit(dataset)
    # dataset[dataset.columns] = imp_mean.transform(dataset)

    dataset[dataset.columns] = mice(dataset.values)
    print("impute finished")

    # dataset = dataset.fillna(0)

    # dataset.to_csv(training_dataset_output_path + 'sleep_model_training_data.csv', index=False)
    dataset = dataset.drop(['timestamp'], axis=1)

    X = dataset.drop(['sleep_or_wake'], axis=1)
    y = dataset['sleep_or_wake']

    # normalizing
    scaler = MinMaxScaler(feature_range=(0, 1))
    X[X.columns] = scaler.fit_transform(X[X.columns])

    return X, y


def get_classifier_name(clf):
    return str(clf.__class__.__name__)


def calculate_feature_importance(X, y):
    model = XGBClassifier()

    print("Fitting " + get_classifier_name(model) + "...")

    model.fit(X, y)

    print("Top 10 most important features:")
    df_feature_imp = pd.DataFrame(model.feature_importances_, index=X.columns,
                                  columns=['feature importance']).sort_values(
        'feature importance', ascending=False)
    print(df_feature_imp.iloc[:10, ])


if __name__ == '__main__':
    X, y = generate_training_data()
    clf = AdaBoostClassifier(n_estimators=100, random_state=0)
    # calculate_feature_importance()
    train_data(clf, X, y)
