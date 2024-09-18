import pandas as pd
import joblib

# Load model
knn_model = joblib.load('knn_model.pkl')

# Data baru dalam format DataFrame dengan nama kolom yang sama
new_data = pd.DataFrame([[5.9, 3.0, 5.1, 1.8]], columns=['sepal_length', 'sepal_width', 'petal_length', 'petal_width'])

# Prediksi dengan model yang disimpan
prediction = knn_model.predict(new_data)
print(f"Prediksi untuk data baru: {prediction}")
