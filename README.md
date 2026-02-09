# Developing a Neural Network Regression Model

## AIM

To develop a neural network regression model for the given dataset.

## THEORY

The objective of this experiment is to develop a neural network regression model using a dataset created in Google Sheets with one numeric input and one numeric output. Regression is a supervised learning technique used to predict continuous values. A neural network is chosen because it can effectively learn both linear and non-linear relationships between input and output by adjusting its weights during training.

The model is trained using backpropagation to minimize a loss function such as Mean Squared Error (MSE). During each iteration, the training loss is calculated and updated. The training loss vs iteration plot is used to visualize the learning process of the model, where a decreasing loss indicates that the neural network is learning properly and converging toward an optimal solution.

## Neural Network Model

<img width="1115" height="695" alt="image" src="https://github.com/user-attachments/assets/1a3163ce-4b0f-4fa5-9b13-61396699d3a9" />


## DESIGN STEPS

### STEP 1:

Loading the dataset

### STEP 2:

Split the dataset into training and testing

### STEP 3:

Create MinMaxScalar objects ,fit the model and transform the data.

### STEP 4:

Build the Neural Network Model and compile the model.

### STEP 5:

Train the model with the training data.

### STEP 6:

Plot the performance plot

### STEP 7:

Evaluate the model with the testing data.

## PROGRAM
### Name: Nandhini M
### Register Number: 212224040211
```python
#creating model class
class NeuralNet(nn.Module):
  def __init__(self):
        super().__init__()
        self.fc1=nn.Linear(1, 8)
        self.fc2=nn.Linear(8, 10)
        self.fc3=nn.Linear(10, 1)
        self.relu=nn.ReLU()
        self.history={'loss':[]}

  def forward(self,x):
        x=self.relu(self.fc1(x))
        x=self.relu(self.fc2(x))
        x=self.fc3(x)
        return x

# Initialize the Model, Loss Function, and Optimizer
ai_brain = NeuralNet()
criterion=nn.MSELoss()
optimizer=optim.RMSprop(ai_brain.parameters(), lr=0.001)

#Function to train model
def train_model(ai_brain, X_train, y_train, criterion, optimizer, epochs=2000):

    for epoch in range(epochs):
      optimizer.zero_grad()
      loss=criterion(ai_brain(X_train),y_train)
      loss.backward()
      optimizer.step()

      ai_brain.history['loss'].append(loss.item())
      if epoch % 200 == 0:
          print(f'Epoch [{epoch}/{epochs}], Loss: {loss.item():.6f}')
```
## Dataset Information

<img width="373" height="418" alt="image" src="https://github.com/user-attachments/assets/202bed6d-2311-4adf-92c2-21bcaa7b48f4" />

## OUTPUT

<img width="722" height="276" alt="image" src="https://github.com/user-attachments/assets/e73d82d9-bfa3-4bd7-af29-e47458644c88" />

<img width="771" height="42" alt="image" src="https://github.com/user-attachments/assets/caa2594e-2edc-4400-9677-0040734cb246" />


### Training Loss Vs Iteration Plot

<img width="688" height="546" alt="image" src="https://github.com/user-attachments/assets/b405788c-6521-4c0f-a8e9-fd48902d122d" />


### New Sample Data Prediction

<img width="466" height="45" alt="image" src="https://github.com/user-attachments/assets/860ddbfc-9a98-4053-aba3-141d3df0cce9" />


## RESULT

Thus the neural network regression model is developed using the given dataset.
