# Heart-Attack-Risk-Prediction-Model-using-TF
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf


df = pd.read_csv('heart_attack_prediction_dataset.csv')


columns_to_keep = ['Age', 'Sex', 'Cholesterol', 'Blood Pressure', 'Smoking', 'Stress Level', 'Heart Attack Risk']
df = df[columns_to_keep]


print(df.isnull().sum())


df.dropna(inplace=True)

label_encoder = LabelEncoder()
df['Sex'] = label_encoder.fit_transform(df['Sex'])
df['Smoking'] = df['Smoking'].astype(int)  # Assuming Smoking is already binary (0/1)
df['Heart Attack Risk'] = df['Heart Attack Risk'].astype(int)  # Assuming the target variable is binary (0/1)


df[['Systolic BP', 'Diastolic BP']] = df['Blood Pressure'].str.split('/', expand=True).astype(int)
df.drop(columns=['Blood Pressure'], inplace=True)

# Normalize/Standardize numerical variables
scaler = StandardScaler()
numerical_features = ['Age', 'Cholesterol', 'Stress Level', 'Systolic BP', 'Diastolic BP']
df[numerical_features] = scaler.fit_transform(df[numerical_features])

# Display the first few rows of the cleaned dataframe
print(df.head())
# Visualization settings
sns.set(style="whitegrid")

# Age vs Heart Attack Risk
plt.figure(figsize=(12, 6))
sns.catplot(x='Heart Attack Risk', y='Age', data=df, palette='Set2', kind="bar")
plt.title('Age vs Heart Attack Risk')
plt.xlabel('Heart Attack Risk')
plt.ylabel('Age')
plt.show()

# Smoking vs Heart Attack Risk
plt.figure(figsize=(12, 6))
sns.countplot(x='Smoking', hue='Heart Attack Risk', data=df, palette='Set2')
plt.title('Smoking vs Heart Attack Risk')
plt.xlabel('Smoking')
plt.ylabel('Count')
plt.legend(title='Heart Attack Risk', loc='upper right')
plt.show()

# Stress Level vs Heart Attack Risk
plt.figure(figsize=(12, 6))
sns.boxplot(x='Heart Attack Risk', y='Stress Level',data=df, palette='Set2')
plt.title('Stress Level vs Heart Attack Risk')
plt.xlabel('Heart Attack Risk')
plt.ylabel('Stress Level')
plt.show()

# Systolic Blood Pressure vs Heart Attack Risk
plt.figure(figsize=(12, 6))
sns.lineplot(x='Heart Attack Risk', y='Systolic BP', data=df, palette='Set2')
plt.title('Systolic Blood Pressure vs Heart Attack Risk')
plt.xlabel('Heart Attack Risk')
plt.ylabel('Systolic Blood Pressure')
plt.show()

# Diastolic Blood Pressure vs Heart Attack Risk
plt.figure(figsize=(12, 6))
sns.lineplot(x='Heart Attack Risk', y='Diastolic BP', data=df, palette='Set2')
plt.title('Diastolic Blood Pressure vs Heart Attack Risk')
plt.xlabel('Heart Attack Risk')
plt.ylabel('Diastolic Blood Pressure')
plt.show()

X = df.drop(columns=['Heart Attack Risk'])
y = df['Heart Attack Risk']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])


model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2)


loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test Accuracy: {accuracy}')
# Function to preprocess user input
def preprocess_input(age, sex, cholesterol, blood_pressure, smoking, stress_level):
    # Encode categorical variables
    sex = label_encoder.transform([sex])[0]
    smoking = int(smoking)

    # Split Blood Pressure into Systolic and Diastolic
    systolic_bp, diastolic_bp = map(int, blood_pressure.split('/'))

    # Normalize numerical variables
    input_data = pd.DataFrame([[age, cholesterol, stress_level, systolic_bp, diastolic_bp]], columns=numerical_features)
    input_data = scaler.transform(input_data)

    # Create a DataFrame with the processed input
    processed_input = pd.DataFrame([[input_data[0][0], sex, input_data[0][1], input_data[0][2], smoking, input_data[0][3], input_data[0][4]]],
                                   columns=X.columns)
    return processed_input

# Function to predict heart attack risk
def predict_heart_attack_risk(age, sex, cholesterol, blood_pressure, smoking, stress_level):
    processed_input = preprocess_input(age, sex, cholesterol, blood_pressure, smoking, stress_level)
    prediction = model.predict(processed_input)
    risk = 'High' if prediction[0][0] >= 0.5 else 'Low'
    return risk

# Example of user input prediction
age = 70
sex = 'Female'
cholesterol = 850
blood_pressure = '140/90'
smoking = 1
stress_level = 9

risk = predict_heart_attack_risk(age, sex, cholesterol, blood_pressure, smoking, stress_level)
print(f'Predicted Heart Attack Risk: {risk}')
