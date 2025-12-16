# src/features/woe_transformer.py
import pandas as pd
from xverse.transformer import WOE

class WoETransformer:
    """
    WoETransformer wraps xverse's WOE for modular use in a pipeline.
    """

    def init(self, categorical_cols):
        """
        Parameters
        ----------
        categorical_cols : list of str
            Columns to transform using WoE.
        """
        self.categorical_cols = categorical_cols
        self.woe_models = {}  # store a WoE model per column

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """
        Fit WoE for each categorical column.

        Parameters
        ----------
        X : pd.DataFrame
            Feature dataframe
        y : pd.Series
            Binary target (0 = good, 1 = bad)
        """
        for col in self.categorical_cols:
            woe = WOE()
            woe.fit(X[[col]], y)
            self.woe_models[col] = woe
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform categorical columns to WoE values.

        Parameters
        ----------
        X : pd.DataFrame
            Feature dataframe

        Returns
        -------
        pd.DataFrame
            Transformed dataframe with WoE values
        """
        X_transformed = X.copy()
        for col in self.categorical_cols:
            woe = self.woe_models[col]
            X_transformed[col] = woe.transform(X[[col]])
        return X_transformed

    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """
        Fit and transform in one step.
        """
        return self.fit(X, y).transform(X)