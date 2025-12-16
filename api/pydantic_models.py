from pydantic import BaseModel
from typing import List

class CustomerData(BaseModel):
    total_amount: float
    avg_amount: float
    transaction_count: int
    std_amount: float
    ProviderId: str
    ProductCategory: str
    ProductId: str
    ChannelId: str

class RiskPrediction(BaseModel):
    probability_high_risk: float
