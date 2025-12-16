# import libraries
import pandas as pd
from sklearn.preprocessing import StandardScaler # for normalizing
from sklearn.cluster import KMeans #Clustering aligorithm



# difine function for target proxy
def create_proxy_target(df,
                        customer_col="CustomerID",
                        datetime_col="TransactionStartTime",
                        amount_col="Amount",
                        n_clusters=3, #number of clusters for kMeans
                        random_state=42):# ensure reproducable clustering results
    df=df.copy()
    # change datetime object to date time format
    df[datetime_col]=pd.to_datetime(df[datetime_col])
    snapshot_date=df[datetime_col].max() + pd.Timedelta(days=1)

    # RFM Calculation
    rfm=(
       df.groupby(customer_col).agg(
         Recency= (
             datetime_col,lambda x:((snapshot_date -x.max()).days) # finds the most recent transaction smallvalue=mostrecent
         ),
         Frequency= (datetime_col,"count"),#Higher value=moreactive customer
         Monetary=(amount_col,"sum") # Higher value=customer spent more money
       ) .reset_index()
    )
    # scale RFM
    sc=StandardScaler()
    rfm_sc=sc.fit_transform(
        rfm[["Recency","Frequency","Monetary"]] # prevents features with large ranges from dominating clustering
    )
    #KMeans
    Kmeans=KMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        n_init=10
    )
    rfm["cluster"]=Kmeans.fit_predict(rfm_sc)
    # identify high cluster
    cluster_profile=(
        rfm.groupby("cluster")[["Frequency","Monetary"]]
        .mean().sort_values(by=["Frequency","Monetary"])
    )
    high_risk_cluster=cluster_profile.index[0]

    rfm["is_high_risk"] = (rfm["cluster"] == high_risk_cluster).astype(int)
    return rfm[[customer_col, "is_high_risk"]]
    




    
                        
                        