# import libraries
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.model_selection import train_test_split,GridSearchCV #for splitting in test and train
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,roc_auc_score 
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier 

#--------------------------------------
  # Class For Training
#--------------------------------------

class ModelTrainer:
    def __init__(self,df:pd.DataFrame,target_col:str,test_size:float=0.2,random_state:int=42):# define variable
        self.df=df
        self.target_col=target_col
        self.test_size=test_size
        self.random_state=random_state
        self.models={}
        self.X_train=self.X_test=self.Y_train=self.Y_test=None
    #-------------------------------------------------------------------
      # Train Test split for modeling
    #----------------------------------------------------------------------
    def prepare_data(self):
        X=self.df.drop(columns=[self.target_col]) # drop the target column to have independent feature only
        Y=self.df[self.target_col] # have the dependent feature alone
        # split the data in to train and test
        self.X_train,self.X_test,self.Y_train,self.Y_test=train_test_split(X,Y,test_size=self.test_size,
                                                                           random_state=self.random_state)
      #------------------------------------------------------------------------
         #add the model
      #---------------------------------------------------------------------------
    def add_model(self,name:str,model):
        """Add model to The Trainer"""
        self.models[name]=model
    #--------------------------------------------------------------------
     # Train the machine with the data set
     # predict and evaluate
    #--------------------------------------------------------------------
    def train_model(self,name:str):
        if name not in self.models:
            raise ValueError(f"model{name}not added")
        model=self.models[name]
        # mlflow experiment tracking
        with mlflow.start_run(run_name=name):
            model.fit(self.X_train,self.Y_train)
            y_pred=model.predict(self.X_test)
            # calculate the metrics
            metrics={
                "accuracy":accuracy_score(self.Y_test,y_pred),
                "precision":precision_score(self.Y_test,y_pred),
                "recall":recall_score(self.Y_test,y_pred),
                "f1_score":f1_score(self.Y_test,y_pred),
                "roc_auc":roc_auc_score(self.Y_test,model.predict_proba(self.X_test)[:,1])
            }
            # log the model and matrix to mlflow
            mlflow.log_params(model.get_params())
            mlflow.log_metrics(metrics)
            # Just log the model inside the run
            mlflow.sklearn.log_model(model, artifact_path=name)

# Or if you want to register it
# mlflow.sklearn.log_model(model, artifact_path="model", registered_model_name=name)

            
            return metrics
    def train_all(self):
         """Train all models and return results"""
         results = {}
         for name in self.models:
            results[name] = self.train_model(name)
         return results
       

    def hyperparameter_tuning(self, name: str, param_grid: dict, cv: int = 5, scoring: str = "roc_auc"):
        """Perform GridSearchCV hyperparameter tuning"""
        if name not in self.models:
            raise ValueError(f"Model {name} not added.")
        
        model = self.models[name]
        grid_search = GridSearchCV(model, param_grid, cv=cv, scoring=scoring, n_jobs=-1)
        grid_search.fit(self.X_train, self.Y_train)
        
        # Update the model with the best estimator
        self.models[name] = grid_search.best_estimator_
        return grid_search.best_params_, grid_search.best_score_

      

        
    
    
                 

