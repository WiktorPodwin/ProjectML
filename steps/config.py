from zenml.steps import BaseParameters

class ModelNameConfig(BaseParameters):
    """
    Model configuration
    """
    # RandomForestClassifier, LogisticRegression, KNeighborsClassifier, SVM, GaussianNB, BaggingClassifier
    name_of_model: str = "RandomForestClassifier"
    fine_tuning: bool = True

class DataTransformConfig(BaseParameters):
    """
    Data transforming configuration
    """
    # PCA, LDA
    name_of_transformation: str = "PCA"
    number_dimensions: int = 9
    fine_transforming: bool = True
