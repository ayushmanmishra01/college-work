from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# 1. Load built-in Iris dataset
iris = load_iris()
X = iris.data    # Features: Sepal & Petal measurements
y = iris.target  # Target labels: 0 (Setosa), 1 (Versicolor), 2 (Virginica)

# 2. Split dataset into Training (80%) and Testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Initialize and train a Decision Tree Classifier
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# 4. Evaluate Model Accuracy
predictions = model.predict(X_test)
print(f"Model Accuracy: {accuracy_score(y_test, predictions) * 100:.2f}%")

# 5. Predict on a new sample flower
# [sepal length, sepal width, petal length, petal width]
new_flower = [[5.1, 3.5, 1.4, 0.2]]
predicted_class = model.predict(new_flower)[0]

print(f"New Flower Features: {new_flower[0]}")
print(f"Predicted Species: {iris.target_names[predicted_class]}")