from __future__ import annotations

from pydantic import BaseModel, Field, model_validator


class ChurnDefinition(BaseModel):
    key: str
    industry: str
    entity: str
    customer_id_column: str
    target_column: str
    positive_event: str
    observation_period: str | None
    prediction_horizon: str | None
    feature_cutoff: str | None
    eligibility: str
    historical_requirements: str
    evaluation_strategy: str
    feature_columns: tuple[str, ...]
    numeric_columns: tuple[str, ...]
    categorical_columns: tuple[str, ...]
    risk_boundaries: tuple[float, float, float] = (0.30, 0.60, 0.80)

    @model_validator(mode="after")
    def validate_columns(self) -> ChurnDefinition:
        if set(self.numeric_columns) | set(self.categorical_columns) != set(self.feature_columns):
            raise ValueError("numeric and categorical columns must cover features exactly")
        if set(self.numeric_columns) & set(self.categorical_columns):
            raise ValueError("a feature cannot be both numeric and categorical")
        a, b, c = self.risk_boundaries
        if not 0 < a < b < c < 1:
            raise ValueError("risk boundaries must be ascending within (0, 1)")
        return self


BANKING = ChurnDefinition(
    key="banking-account-exit-v1", industry="banking", entity="customer",
    customer_id_column="CustomerId", target_column="Exited",
    positive_event="Exited=1: customer exited the bank relationship in the source snapshot",
    observation_period=None, prediction_horizon=None, feature_cutoff=None,
    eligibility="One unique customer record with valid required features and Exited in {0,1}",
    historical_requirements="One labeled cross-sectional snapshot; no temporal history supplied",
    evaluation_strategy="Stratified random holdout; no temporal or prospective claims",
    feature_columns=("CreditScore", "Geography", "Gender", "Age", "Tenure", "Balance",
                     "NumOfProducts", "HasCrCard", "IsActiveMember", "EstimatedSalary"),
    numeric_columns=("CreditScore", "Age", "Tenure", "Balance", "NumOfProducts",
                     "HasCrCard", "IsActiveMember", "EstimatedSalary"),
    categorical_columns=("Geography", "Gender"),
)


class RiskBand(BaseModel):
    name: str
    minimum: float = Field(ge=0, le=1)


def risk_band(probability: float, definition: ChurnDefinition = BANKING) -> str:
    low, medium, high = definition.risk_boundaries
    if probability < low:
        return "low"
    if probability < medium:
        return "medium"
    if probability < high:
        return "high"
    return "critical"

