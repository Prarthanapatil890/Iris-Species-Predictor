# Import necessary libraries
from sklearn.datasets import load_iris                 # To load the built-in Iris dataset
from sklearn.ensemble import RandomForestClassifier    # Random Forest model for classification
import joblib                                          # To save the trained model to disk


# Step 1: Load the Iris dataset

iris = load_iris()                                       # Loads the Iris dataset with data and target labels
X, y = iris.data, iris.target                            # X = features (sepal/petal measurements), y = labels (species)

# Step 2: Train the Random Forest Classifier

model = RandomForestClassifier()                          # Create an instance of the Random Forest Classifier
model.fit(X, y)                                           # Train the model using the entire dataset (X, y)

# Step 3: Save the trained model to a file
# This saves the model to a file named 'iris_model.pkl'
joblib.dump(model, "iris_model.pkl")

# Output:
# A file named 'iris_model.pkl' will be created in your working directory.
# You can load it using: model = joblib.load("iris_model.pkl")
