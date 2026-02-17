# Developing a Neural Network Regression Model

## AIM

To develop a neural network regression model for the given dataset.

## THEORY

The objective of this experiment is to develop a neural network regression model using a dataset created in Google Sheets with one numeric input and one numeric output. Regression is a supervised learning technique used to predict continuous values. A neural network is chosen because it can effectively learn both linear and non-linear relationships between input and output by adjusting its weights during training.

The model is trained using backpropagation to minimize a loss function such as Mean Squared Error (MSE). During each iteration, the training loss is calculated and updated. The training loss vs iteration plot is used to visualize the learning process of the model, where a decreasing loss indicates that the neural network is learning properly and converging toward an optimal solution.

## Neural Network Model

<img width="770" height="644" alt="image" src="https://github.com/user-attachments/assets/3f5325e6-eba1-405c-8e8f-6c0c27589abf" />


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
class NeuralNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1, 8)
        self.fc2 = nn.Linear(8, 10)
        self.fc3 = nn.Linear(10, 1)
        self.relu = nn.ReLU()
        self.history = {'loss': []}

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

nandhini = NeuralNet()
criterion = nn.MSELoss()
optimizer = optim.RMSprop(nandhini.parameters(), lr=0.001)

X_train = torch.tensor([[1.0],[2.0],[3.0],[4.0],[5.0]])
y_train = torch.tensor([[2.0],[4.0],[6.0],[8.0],[10.0]])

X_test = torch.tensor([[6.0],[7.0],[8.0]])
y_test = torch.tensor([[12.0],[14.0],[16.0]])

def train_model(model, X, y, criterion, optimizer, epochs=2000):
    for epoch in range(epochs):
        optimizer.zero_grad()
        output = model(X)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()

        model.history['loss'].append(loss.item())

        if epoch % 200 == 0:
            print(f'Epoch [{epoch}/{epochs}] Loss: {loss.item():.6f}')

train_model(nandhini, X_train, y_train, criterion, optimizer)

nandhini.eval()

with torch.no_grad():
    test_predictions = nandhini(X_test)
    test_loss = criterion(test_predictions, y_test)

print(f"\nTest Loss: {test_loss.item():.6f}")

print("\nPredictions vs Actual:")
for pred, actual in zip(test_predictions, y_test):
    print(f"{pred.item():.2f}  vs  {actual.item()}")

test_value = torch.tensor([[10.0]])   # change to any value you want
print("\nPrediction:", nandhini(test_value).item())

plt.plot(nandhini.history['loss'])
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss")
plt.show()

```
## Dataset Information

<img width="373" height="418" alt="image" src="https://github.com/user-attachments/assets/202bed6d-2311-4adf-92c2-21bcaa7b48f4" />

## OUTPUT

<img width="290" height="214" alt="image" src="https://github.com/user-attachments/assets/e5eb2981-be8b-4ad8-b4ff-9d4ccb59b0d2" />


<img width="181" height="83" alt="image" src="https://github.com/user-attachments/assets/0b8e0ff6-b0c8-41a6-84b9-c7ca651e9403" />


### Training Loss Vs Iteration Plot

<img width="560" height="452" alt="image" src="https://github.com/user-attachments/assets/4fc26224-3e33-485e-8db9-c21b3d520774" />


### New Sample Data Prediction

<img width="237" height="23" alt="image" src="https://github.com/user-attachments/assets/b6b41a46-9c08-468d-93ee-1093401026e8" />


## RESULT

Thus the neural network regression model is developed using the given dataset.
