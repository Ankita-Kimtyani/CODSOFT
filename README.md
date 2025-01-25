# Titanic Survival Model Project

## Overview

Welcome to the Titanic Survival Model project! This project was developed during my internship at Codsoft. The main goal of this project is to create a predictive model that determines the survival chances of passengers on the RMS Titanic based on various factors such as age, gender, ticket class, etc.

## Table of Contents

1. [Introduction](#introduction)
2. [Installation](#installation)
3. [Dataset](#dataset)
4. [Data Preprocessing](#data-preprocessing)
5. [Modeling](#modeling)
6. [Evaluation](#evaluation)
7. [Usage](#usage)
8. [Conclusion](#conclusion)
9. [Acknowledgments](#acknowledgments)

## Introduction

The sinking of the RMS Titanic is one of the most infamous shipwrecks in history. While exploring the dataset, the aim is to create a model that predicts the survival of passengers. This project leverages machine learning techniques, specifically a neural network, to achieve this goal.

## Installation

To run this project, you will need to have the following installed:

- Python 3.x
- Jupyter Notebook
- Required libraries: numpy, pandas, tensorflow, scikit-learn

You can install the required libraries using the following command:

```bash
pip install numpy pandas tensorflow scikit-learn
```

## Dataset

The dataset used in this project is provided by Kaggle which is attached with files .

## Data Preprocessing

The data preprocessing steps involve:

1. Converting categorical variables ('Sex' and 'Embarked') to numerical values.
2. Filling missing values in 'Age', 'Embarked', and 'Fare' with their respective mean or mode.
3. Selecting relevant features for the model.
4. Splitting the data into training and testing sets.
5. Standardizing the features.

Here's a snippet of the preprocessing code:

```python
def preprocess_titanic_data(dataframe):
    dataframe['Sex'] = dataframe['Sex'].map({'male': 0, 'female': 1})
    dataframe['Embarked'] = dataframe['Embarked'].map({'C': 0, 'Q': 1, 'S': 2})
    dataframe['Age'] = dataframe['Age'].fillna(dataframe['Age'].mean())
    dataframe['Embarked'] = dataframe['Embarked'].fillna(dataframe['Embarked'].mode()[0])
    dataframe['Fare'] = dataframe['Fare'].fillna(dataframe['Fare'].mean())
    features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
    X = dataframe[features]
    y = dataframe['Survived'] if 'Survived' in dataframe else None
    return X, y
```

## Modeling

The neural network model consists of:

1. Input layer with 16 neurons and ReLU activation.
2. Hidden layer with 8 neurons and ReLU activation.
3. Output layer with 1 neuron and sigmoid activation.

The model is compiled using the Adam optimizer and binary cross-entropy loss. It is trained over 10 epochs with a batch size of 32.

Here's a snippet of the model creation code:

```python
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Build the neural network model
model = Sequential()
model.add(Dense(16, input_dim=X_train.shape[1], activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

```

## Evaluation

The model's performance is evaluated using accuracy, precision, recall, and F1 score. The results indicate the effectiveness of the neural network in predicting survival.

## Usage

You can use the trained model to predict the survival chances of new passengers. Here is an example of predicting survival probabilities for sample data:

```python
sample_data = [
    [3, 'female', 30, 1, 0, 7.25, 'S'],
    [1, 'male', 40, 0, 0, 100, 'C']
]
sample_dataframe = pd.DataFrame(sample_data, columns=['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked'])
X_sample, _ = preprocess_titanic_data(sample_dataframe)
X_sample = scaler.transform(X_sample)
pred = model.predict(X_sample)

for i, prediction in enumerate(pred):
    print(f"Sample {i+1}'s Surviving Rate: {prediction}")
```

## Conclusion

The project successfully developed a predictive model to determine the survival chances of Titanic passengers. The insights gained from this project can be applied to similar predictive modeling problems in the future.

## Acknowledgments

I would like to express my gratitude to Codsoft for providing the opportunity to work on this project.
