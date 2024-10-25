############ Script for Training Binary Classification Models based on SkLearn #############

from abc import ABC, abstractmethod
from typing import List
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class DataLoader(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def clean_data(self):
        pass

    @abstractmethod
    def get_splits(self):
        pass


class TelcoDataLoader(DataLoader):
    def __init__(self, data_path: str):
        super(TelcoDataLoader, self).__init__()
        self.df = pd.read_excel(data_path)

    def clean_data(self):
        """
        1. Drops irrelevant columns
        2. Handles Numeric and Categorical Features
        3. This is based on study, will need more automation
        """
        irrelevant_columns = [
            "Latitude",
            "Longitude",
            "CustomerID",
            "Count",
            "Country",
            "State",
            "City",
            "Zip Code",
            "Lat Long",
            "Churn Reason",
            "Churn Label",
            "CLTV",
            "Churn Score",
        ]

        self.df = self.df.drop(irrelevant_columns, axis=1, errors="ignore")
        no_internet_columns = [
            "Online Security",
            "Online Backup",
            "Device Protection",
            "Tech Support",
            "Streaming TV",
            "Streaming Movies",
        ]  # all service columns

        for feature in no_internet_columns:
            self.df[feature] = self.df[feature].map(
                {"No": "No", "Yes": "Yes", "No internet service": "No"}
            )

        self.df["Multiple Lines"] = self.df["Multiple Lines"].map(
            {"No": "No", "Yes": "Yes", "No phone service": "No"}
        )
        self.df["Gender"] = self.df["Gender"].map({"Male": "Yes", "Female": "No"})
        df_enc = pd.get_dummies(
            self.df, columns=["Contract", "Internet Service", "Payment Method"]
        )
        for col in list(df_enc.columns):
            if col in [
                "Tenure Months",
                "Monthly Charges",
                "Total Charges",
                "Churn Value",
            ]:
                continue
            if df_enc[col].dtype == "bool":
                df_enc[col] = df_enc[col].map({True: 1, False: 0})
            else:
                df_enc[col] = df_enc[col].map({"Yes": 1, "No": 0})
        df_enc["Total Charges"] = pd.to_numeric(
            df_enc["Total Charges"], errors="coerce"
        )
        null_mask = df_enc.isnull().any(axis=1)
        null_rows = df_enc[null_mask].index
        df_enc.drop(index=null_rows, inplace=True, errors="raise")
        df_enc = df_enc.reset_index(drop=True)

        self.df = df_enc

    def feature_reduction(self, features_to_select: List):

        ### numeric features scaling ###
        self.df_label = self.df["Churn Value"]
        self.df = self.df.drop(["Churn Value"], axis=1)
        numeric_features = ["Tenure Months", "Monthly Charges", "Total Charges"]
        df_numeric = self.df[numeric_features]
        self.df = self.df.drop(numeric_features, axis=1)
        scaler = StandardScaler()
        df_numeric = pd.DataFrame(
            scaler.fit_transform(df_numeric), columns=numeric_features
        )
        self.df = pd.concat([self.df, df_numeric], axis=1)

        ### selection ###
        self.df = self.df[features_to_select]

    def get_splits(self, test_size=0.2, seed=42):
        X_train, X_test, y_train, y_test = train_test_split(
            self.df, self.df_label, test_size=test_size, random_state=seed
        )
        return {
            "X_train": X_train,
            "y_train": y_train,
            "X_test": X_test,
            "y_test": y_test,
        }
