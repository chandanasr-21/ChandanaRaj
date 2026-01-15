"""
Classification Example:
Predict whether a student will PASS or FAIL based on study hours.
"""

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# Sample dataset
# Study hours vs Pass(1) / Fail(0)
X = [[1], [2], [3], [4], [5], [6], [7], [8]]
y = [0, 0, 0, 0, 1, 1, 1, 1]

# Split data into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# Create and train model
model = LogisticRegression()
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("Classification Model Accuracy:", accuracy)

# Predict new data
study_hours = [[6]]
result = model.predict(study_hours)

if result[0] == 1:
    print("Prediction: Student will PASS")
else:
    print("Prediction: Student will FAIL")
