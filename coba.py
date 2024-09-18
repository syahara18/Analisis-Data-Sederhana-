import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Menambahkan nama kolom sesuai dataset Iris
col_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']
df = pd.read_csv('C:/Users/ARA/Desktop/iris.csv', header=None, names=col_names)

# Melihat 5 baris pertama dan informasi dataset
print(df.head())  
print(df.info())  
print(df.describe())

# Visualisasi histogram sepal_length
sns.histplot(df['sepal_length'], kde=True)
plt.title('Distribution of Sepal Length')
plt.xlabel('Sepal Length')
plt.ylabel('Frequency')
plt.show()

# Visualisasi scatter plot sepal_length vs sepal_width
sns.scatterplot(x='sepal_length', y='sepal_width', data=df)
plt.title('Sepal Length vs Sepal Width')
plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')
plt.show()

# Menghitung korelasi hanya pada kolom numerik
corr_matrix = df.drop('species', axis=1).corr()

# Membuat heatmap korelasi
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix of Iris Dataset')
plt.show()

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Memisahkan fitur dan label
X = df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
y = df['species']

# Membagi data menjadi training dan testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Membuat model KNN
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

# Memprediksi data testing
y_pred = knn.predict(X_test)

# Menghitung akurasi model
accuracy = accuracy_score(y_test, y_pred)
print(f'Akurasi model KNN: {accuracy * 100:.2f}%')

from sklearn.metrics import confusion_matrix, classification_report

# Menghitung confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print('Confusion Matrix:')
print(conf_matrix)

# Menghitung classification report
class_report = classification_report(y_test, y_pred)
print('Classification Report:')
print(class_report)

import joblib

# Menyimpan model
joblib.dump(knn, 'knn_model.pkl')
print('Model saved as knn_model.pkl')

# Memuat model
loaded_model = joblib.load('knn_model.pkl')

# Menggunakan model untuk prediksi
new_predictions = loaded_model.predict(X_test)

print('Prediksi untuk data testing baru:', new_predictions)


