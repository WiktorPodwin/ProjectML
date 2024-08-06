from zenml.steps import BaseParameters

class ModelNameConfig(BaseParameters):
    """
    Model configuration
    """
    # RandomForestClassifier, LogisticRegression, KNeighborsClassifier, SVM, GaussianNB
    name_of_model: str = "GaussianNB"
    fine_tuning: bool = False

class DataTransformConfig(BaseParameters):
    """
    Data transforming configuration
    """
    # PCA, LDA
    name_of_transformation: str = "LDA"
    number_dimensions: int = 6
    fine_transforming: bool = False
