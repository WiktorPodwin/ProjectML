from zenml.steps import BaseParameters

class ModelNameConfig(BaseParameters):
    """
    Model configuration
    """
    name_of_model: str = "RandomForestClassifier"
    fine_tuning: bool = False
