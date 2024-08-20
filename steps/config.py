from zenml.steps import BaseParameters

class ModelNameConfig(BaseParameters):
    """
    Model configuration
    """
    # RandomForestClassifier, LogisticRegression, KNeighborsClassifier, SVM, GaussianNB, BaggingClassifier, TensorflowNeuralNetwork, PyTorchNeuralNetwork
    name_of_model: str = "RandomForestClassifier"
    fine_tuning: bool = False

class DataTransformConfig(BaseParameters):
    """
    Data transforming configuration
    """
    # PCA, LDA
    name_of_transformation: str = "PCA"
    number_dimensions: int = 10
    fine_transforming: bool = False
