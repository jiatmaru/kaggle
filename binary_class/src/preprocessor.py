import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder

class CustomPreprocessor:
    def __init__(self):
        self.age_max = None
        self.min_balance = None
        self.marital_map = {"married": 1, "single": 0}
        self.binary_map = {"no": 0, "yes": 1}
        self.month_map = {'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6,
                          'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12}
        
        self.poutcome_ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        self.job_ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        self.education_ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        self.contact_ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

    def fit(self, df):
        if 'age' in df.columns:
            self.age_max = df['age'].max()
        if 'balance' in df.columns:
            self.min_balance = df['balance'].min()

        if 'poutcome' in df.columns:
            self.poutcome_ohe.fit(df[['poutcome']])
        if 'job' in df.columns:
            self.job_ohe.fit(df[['job']])
        if 'education' in df.columns:
            self.education_ohe.fit(df[['education']])
        if 'contact' in df.columns:
            self.contact_ohe.fit(df[['contact']])
        
        return self

    def transform(self, df):
        df_transformed = df.copy()

        if 'age' in df_transformed.columns:
            df_transformed['age_normalized'] = df_transformed['age'] / self.age_max
            df_transformed = df_transformed.drop('age', axis=1)

        if 'marital' in df_transformed.columns:
            df_transformed['marital_encoded'] = df_transformed['marital'].map(self.marital_map)
            df_transformed = df_transformed.drop('marital', axis=1)

        if 'poutcome' in df_transformed.columns:
            poutcome_encoded = self.poutcome_ohe.transform(df_transformed[['poutcome']])
            poutcome_cols = [f'poutcome_{cat}' for cat in self.poutcome_ohe.categories_[0]]
            df_transformed[poutcome_cols] = poutcome_encoded
            df_transformed = df_transformed.drop('poutcome', axis=1)

        if 'job' in df_transformed.columns:
            job_encoded = self.job_ohe.transform(df_transformed[['job']])
            job_cols = [f'job_{cat}' for cat in self.job_ohe.categories_[0]]
            df_transformed[job_cols] = job_encoded
            df_transformed = df_transformed.drop('job', axis=1)

        if 'education' in df_transformed.columns:
            education_encoded = self.education_ohe.transform(df_transformed[['education']])
            education_cols = [f'education_{cat}' for cat in self.education_ohe.categories_[0]]
            df_transformed[education_cols] = education_encoded
            df_transformed = df_transformed.drop('education', axis=1)
        
        if 'contact' in df_transformed.columns:
            contact_encoded = self.contact_ohe.transform(df_transformed[['contact']])
            contact_cols = [f'contact_{cat}' for cat in self.contact_ohe.categories_[0]]
            df_transformed[contact_cols] = contact_encoded
            df_transformed = df_transformed.drop('contact', axis=1)
        
        if 'default' in df_transformed.columns:
            df_transformed['default_encoded'] = df_transformed['default'].map(self.binary_map)
            df_transformed = df_transformed.drop('default', axis=1)

        if 'balance' in df_transformed.columns:
            if self.min_balance < 0:
                df_transformed['balance_log'] = np.log1p(df_transformed['balance'] - self.min_balance)
            else:
                df_transformed['balance_log'] = np.log1p(df_transformed['balance'])
            df_transformed = df_transformed.drop('balance', axis=1)
        
        for col in ['housing', 'loan']:
            if col in df_transformed.columns:
                df_transformed[f'{col}_encoded'] = df_transformed[col].map(self.binary_map)
                df_transformed = df_transformed.drop(col, axis=1)
        
        if 'day' in df_transformed.columns and 'month' in df_transformed.columns:
            df_transformed['day_sin'] = np.sin(2 * np.pi * df_transformed['day'] / 31)
            df_transformed['day_cos'] = np.cos(2 * np.pi * df_transformed['day'] / 31)
            df_transformed['month'] = df_transformed['month'].map(self.month_map)
            df_transformed['month_sin'] = np.sin(2 * np.pi * df_transformed['month'] / 12)
            df_transformed['month_cos'] = np.cos(2 * np.pi * df_transformed['month'] / 12)
            df_transformed = df_transformed.drop(['day', 'month'], axis=1)
        
        if 'duration' in df_transformed.columns:
            df_transformed['duration_log'] = np.log1p(df_transformed['duration'])
            df_transformed = df_transformed.drop('duration', axis=1)
        
        if 'campaign' in df_transformed.columns:
            df_transformed['campaign_log'] = np.log1p(df_transformed['campaign'])
            df_transformed = df_transformed.drop('campaign', axis=1)

        if 'pdays' in df_transformed.columns:
            df_transformed['pdays_contacted'] = np.where(df_transformed['pdays'] == -1, 0, 1)
            df_transformed['pdays_log'] = np.where(df_transformed['pdays'] == -1, 0, np.log1p(df_transformed['pdays']))
            df_transformed = df_transformed.drop('pdays', axis=1)

        if 'previous' in df_transformed.columns:
            df_transformed['previous_log'] = np.log1p(df_transformed['previous'])
            df_transformed = df_transformed.drop('previous', axis=1)

        return df_transformed
        
    def fit_transform(self, df):
        return self.fit(df).transform(df)