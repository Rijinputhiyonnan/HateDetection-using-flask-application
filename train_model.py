


import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer

# Load your dataset
data = pd.read_excel('labeled_comments.xlsx')  
print(data.head())

train_data = data['cleaned_comments']  # Your text data
y_train = data['label']     # Your labels

# Train the model
vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(train_data)  # Transform the text data to feature vectors

# Initialize the model correctly
model = RandomForestClassifier(
    criterion='gini',
    max_depth=None,
    max_features='log2',
    n_estimators=300,
    random_state=42
)
model.fit(X_train, y_train)  # Fit the model

# Save both model and vectorizer
joblib.dump(model, 'model.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')
