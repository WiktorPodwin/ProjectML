from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
import pandas as pd
import logging
import torch
import torch.nn as nn
import torch.optim as optim 
import numpy as np

class TensorflowNeuralNetworkModel():
    """
     Tensorflow Neural Network class
    """
    def model_train(self, X_train: pd.DataFrame, y_train: pd.Series) -> Sequential:
        """
        Trains the model
        
        Args:
            X_train: Training data
            y_train: Training labels
        Returns:
            Sequential: Trained neural newtork
        """
        try:
            X_train = X_train.to_numpy()
            y_train = y_train.to_numpy()
            model = Sequential()
            model.add(Input(shape=(X_train.shape[1], )))
            model.add(Dense(units=32, activation='relu'))
            model.add(Dense(units=64, activation='relu'))
            model.add(Dense(units=1, activation='sigmoid'))
            model.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy'])
            model.fit(X_train, y_train, epochs=80, batch_size=32)
            return model
        except Exception as e:
            logging.error(f"Error in model training: {e}")
            raise e

class PyTorchNeuralNetworkModel(nn.Module):
    """
    PyTorch Neural Network class
    """
    def __init__(self, input_dimensions: int) -> None:
        """
        Initializes the PyTorchNeuralNetworkModel with specified input dimensions
        
        Args:
            input_dimensions: Number of input features to the model
        """
        super(PyTorchNeuralNetworkModel, self).__init__()
        self.fc1 = nn.Linear(input_dimensions, 32)
        self.fc2 = nn.Linear(32, 64)
        self.fc3 = nn.Linear(64, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs a forward pass through the network
        
        Args:
            x: Input tensor
        Returns:
            torch.Tensor: Output tensor containing probabilities
        """
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x

    def model_train(self, X_train: pd.DataFrame, y_train: pd.Series) -> nn.Module:
        """
        Trains the neural network on a provided training data

        Args:
            X_train: Training data
            y_train: Training labels
        Returns:
            nn.Module: The trained model
        """
        try:
            X_train_tensor = torch.tensor(X_train.to_numpy(), dtype=torch.float32)
            y_train_tensor = torch.tensor(y_train.to_numpy(), dtype=torch.float32).view(-1, 1)            
            criterion = nn.BCELoss()
            optimizer = optim.SGD(self.parameters(), lr=0.01)
            num_epochs = 80
            batch_size = 32
            dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

            for epoch in range(num_epochs):
                self.train()
                running_loss = 0
                correct_predictions = 0
                total_samples = 0
                for inputs, labels in dataloader:
                    optimizer.zero_grad()
                    outputs = self(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()

                    running_loss += loss.item()
                    predicted_classes = (outputs > 0.5).float()
                    correct_predictions += (predicted_classes == labels).sum().item()
                    total_samples += labels.size(0)

                epoch_loss = running_loss / len(dataloader)
                accuracy = (correct_predictions / total_samples)    
                print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}, Accuracy: {accuracy:.2f}")
            return self
        except Exception as e:
            logging.error(f"Error in model training: {e}")
            raise e
    
def pytorch_prediction(model: nn.Module, X_test: pd.DataFrame) -> np.ndarray:
    """
    Calculates predictions on a test dataset
    
    Args:
        model: The trained model for making predictions
        X_test: Test data
    Returns:
        np.ndarray: The predicted class probabilities for each sample in the test dataset
    """
    model.eval()
    with torch.no_grad():
        predictions = model(torch.tensor(X_test.to_numpy(), dtype=torch.float32)).numpy()
    return predictions
