import logging
import os
from glob import glob
from datetime import datetime

from airflow.decorators import task, dag, task_group
import numpy as np
import pandas as pd

from joblib import dump, load
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB

from docs import dag1_doc


def one_hot_encoder(df, nan_as_category=False):
    original_columns = list(df.columns)
    categorical_columns = [col for col in df.columns if df[col].dtype == 'object']
    df = pd.get_dummies(df, columns=categorical_columns, dummy_na=nan_as_category, drop_first=True)
    new_columns = [c for c in df.columns if c not in original_columns]
    return df, new_columns


@dag(dag_id="Test", start_date=datetime(2022, 7, 23), schedule_interval="@hourly", catchup=False,
     doc_md=dag1_doc.doc)
def my_dag():
    @task
    def build_env():
        os.makedirs("/bucket/Test/split_data", exist_ok=True)
        os.makedirs("/bucket/Test/models", exist_ok=True)
        os.makedirs("/bucket/Test/predictions", exist_ok=True)
        os.makedirs("/bucket/Test/results", exist_ok=True)

    @task
    def extract_and_processing():
        # logging.info(df)
        df_credit = pd.read_csv("/bucket/data/german_credit_data.csv", index_col=0)
        logging.info("Creating an categorical variable to handle with the Age variable")
        interval = (18, 25, 35, 60, 120)

        cats = ['Student', 'Young', 'Adult', 'Senior']
        df_credit["Age_cat"] = pd.cut(df_credit.Age, interval, labels=cats)

        df_good = df_credit[df_credit["Risk"] == 'good']
        df_bad = df_credit[df_credit["Risk"] == 'bad']

        df_credit['Saving accounts'] = df_credit['Saving accounts'].fillna('no_inf')
        df_credit['Checking account'] = df_credit['Checking account'].fillna('no_inf')

        # Purpose to Dummies Variable
        df_credit = df_credit.merge(pd.get_dummies(df_credit.Purpose, drop_first=True, prefix='Purpose'),
                                    left_index=True, right_index=True)
        # Sex feature in dummies
        df_credit = df_credit.merge(pd.get_dummies(df_credit.Sex, drop_first=True, prefix='Sex'), left_index=True,
                                    right_index=True)
        # Housing get dummies
        df_credit = df_credit.merge(pd.get_dummies(df_credit.Housing, drop_first=True, prefix='Housing'),
                                    left_index=True, right_index=True)
        # Housing get Saving Accounts
        df_credit = df_credit.merge(pd.get_dummies(df_credit["Saving accounts"], drop_first=True, prefix='Savings'),
                                    left_index=True, right_index=True)
        # Housing get Risk
        df_credit = df_credit.merge(pd.get_dummies(df_credit.Risk, prefix='Risk'), left_index=True,
                                    right_index=True)
        # Housing get Checking Account
        df_credit = df_credit.merge(pd.get_dummies(df_credit["Checking account"], drop_first=True, prefix='Check'),
                                    left_index=True, right_index=True)
        # Housing get Age categorical
        df_credit = df_credit.merge(pd.get_dummies(df_credit["Age_cat"], drop_first=True, prefix='Age_cat'),
                                    left_index=True, right_index=True)
        # Excluding the missing columns
        del df_credit["Saving accounts"]
        del df_credit["Checking account"]
        del df_credit["Purpose"]
        del df_credit["Sex"]
        del df_credit["Housing"]
        del df_credit["Age_cat"]
        del df_credit["Risk"]
        del df_credit['Risk_good']

        df_credit['Credit amount'] = np.log(df_credit['Credit amount'])
        # Creating the X and y variables
        X = df_credit.drop('Risk_bad', 1).values
        y = df_credit["Risk_bad"].values

        # Spliting X and y into train and test version
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

        np.save("/bucket/Test/split_data/X_full", X)
        np.save("/bucket/Test/split_data/y_full", y)
        np.save("/bucket/Test/split_data/X_train", X_train)
        np.save("/bucket/Test/split_data/X_test", X_test)
        np.save("/bucket/Test/split_data/y_train", y_train)
        np.save("/bucket/Test/split_data/y_test", y_test)

    @task
    def train_model1():
        output_file = '/bucket/Test/models/model1.joblib'
        X_train = np.load("/bucket/Test/split_data/X_train.npy")
        # X_test = np.load("/bucket/Test/split_data/X_test.npy")
        y_train = np.load("/bucket/Test/split_data/y_train.npy")
        # y_test = np.load("/bucket/Test/split_data/y_test.npy")

        # Seting the Hyper Parameters
        param_grid = {"max_depth": [3, 5, 7, 10, None],
                      "n_estimators": [3, 5, 10, 25, 50, 150],
                      "max_features": [4, 7, 15, 20]}

        # Creating the classifier
        model = RandomForestClassifier(random_state=2)

        grid_search = GridSearchCV(model, param_grid=param_grid, cv=5, scoring='recall', verbose=4)
        grid_search.fit(X_train, y_train)
        rf = RandomForestClassifier(max_depth=None, max_features=10, n_estimators=15, random_state=2)

        # trainning with the best params
        rf.fit(X_train, y_train)

        dump(rf, output_file)

    @task
    def train_model2():
        output_file = '/bucket/Test/models/model2.joblib'
        X_train = np.load("/bucket/Test/split_data/X_train.npy")
        # X_test = np.load("/bucket/Test/split_data/X_test.npy")
        y_train = np.load("/bucket/Test/split_data/y_train.npy")
        # y_test = np.load("/bucket/Test/split_data/y_test.npy")

        # Criando o classificador logreg

        GNB = GaussianNB()

        # Fitting with train data
        model = GNB.fit(X_train, y_train)
        dump(model, output_file)

    @task
    def predict():
        model_paths = glob("/bucket/Test/*.joblib")
        X = np.load("/bucket/Test/split_data/X_full.npy")
        y = np.load("/bucket/Test/split_data/y_full.npy")

        for model_path in model_paths:
            logging.info(model_path)
            modal = load(model_path)
            y_pred = modal.predict(X)
            basename = os.path.basename(model_path).replace(".joblib", "")
            np.save(f"/bucket/Test/predictions/{basename}_results", y_pred)

    @task
    def save_dataframe():
        result_paths = glob("/bucket/Test/predictions/*.npy")
        for result_path in result_paths:
            basename = os.path.basename(result_path).replace("_results.npy", "")
            df = pd.read_csv("/bucket/data/german_credit_data.csv", index_col=0)
            # X = np.load("/bucket/Test/split_data/X_full.npy")
            # y = np.load("/bucket/Test/split_data/y_full.npy")
            y_predict = np.load(result_path)
            df["y_predict"] = y_predict
            # y = y.reshape((-1, 1))
            # y_predict = y_predict.reshape((-1, 1))
            # logging.info(f"X {X.shape}")
            # logging.info(f"y {y.shape}")
            # logging.info(f"y_predict {y_predict.shape}")

            # final_result = np.concatenate((X, y, y_predict), axis=1)
            # logging.info(f"final_result {final_result.shape}")

            # np.save(f"/bucket/Test/results/{basename}_final_results", final_result)
            # df = pd.DataFrame(data=final_result)
            df.to_csv(f"/bucket/Test/results/{basename}_final_results.csv", index=False)

    build_env() >> extract_and_processing() >> [train_model1(), train_model2()] >> predict() >> save_dataframe()


dag = my_dag()
