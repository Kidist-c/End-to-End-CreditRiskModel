import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator,TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer #applies to different transformer to diferent column
from sklearn.preprocessing import OneHotEncoder,StandardScaler #converts categorical variables into a binary encoded format
from sklearn.impute import SimpleImputer # Handles missing values in datasets

#------------------------------------------
 #Custom Transformer:Date Feature Extraction
#-------------------------------------------
class DateTimeFeature(BaseEstimator,TransformerMixin):
    def __init__(self,datatime_col):
        self.datetime_col=datatime_col
    def fit(self,X,y=None):# fit method whch is required by sklearn
        return self
    def transform(self,X):# applies the main transformation
        X=X.copy()
        X[self.datetime_col]=pd.to_datetime(X[self.datetime_col]) # extract the column related with time
        X["transaction_hour"]=X[self.datetime_col].dt.hour # extract the hour
        X["transaction_day"]=X[self.datetime_col].dt.day # extract the day
        X["transaction_month"]=X[self.datetime_col].dt.month
        X["transaction_year"]=X[self.datetime_col].dt.year


        return X.drop(columns=[self.datetime_col])  # return the updated dataset
#-------------------------------------------------------
 # Custom Transformer:Customer- Level Aggregation
# ------------------------------------------------------
class customerAggregation(BaseEstimator,TransformerMixin):

    def __init__(self,customer_col,amount_col): # the constructor method
        self.customer_col=customer_col
        self.amount_col=amount_col
    def fit(self,X,y=None): # fit method  which expected by sklearn
        return self
    # The main main Transformer method applies the Transformation
    def transform(self,X):
        agg_df=(
            X.groupby(self.customer_col)[self.amount_col].agg(
                total_amount="sum",
                avg_amount="mean",
                transaction_count="count",
                std_amount="std"

            ).reset_index()
        )
            
        
        return X.merge(agg_df, on=self.customer_col, how="left")
# define features
numerical_cols=[
    "total_amount",
    "avg_amount",
    "transaction_count",
    "std_amount"
] # numerical columns for numerical preprocessing
catagorical_cols=[
    "ProviderId",
    "ProductCategory",
    "ProductId",
    "ChannelId"

] #catagorical columns for catagorical analysis

#-----------------------------------------
  # numerical pipeline
#-----------------------------------------

    

num_pipeline=Pipeline(steps=[
    ("imputer",SimpleImputer(strategy="median")),# imputes missing value if available with median
    ("scaler",StandardScaler()) # normalize the numerical columns 
     
])
#---------------------------------------------
 #catagorical pipeleine
#--------------------------------------------
cat_pipeline=Pipeline(steps=[
    ("imputer",SimpleImputer(strategy="most_frequent")),# imputes missing value with most_frequent value
    ("encoder",OneHotEncoder(handle_unknown="ignore",sparse_output=False)) # change catagorical to number

])

#---------------------------------------------------------
  #Combine Pipeline using column Transformer
# -----------------------------------------------------
preprocessor=ColumnTransformer(
    transformers=[
        ("num",num_pipeline,numerical_cols),
        ("cat",cat_pipeline,catagorical_cols)
    ]
)
#-----------------------------------------------
   #Full Feature Engineering Pipeline
#---------------------------------------------
feature_pipeline = Pipeline(steps=[
    ("datetime", DateTimeFeature("TransactionStartTime")),
    ("aggregation", customerAggregation("CustomerId", "Amount")),
    ("preprocess", preprocessor)
])


