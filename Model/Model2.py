# Importing the libraries
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split  # for splitting the data
import numpy as np  # for array operations
import pandas as pd  # for working with DataFrames
import matplotlib.pyplot as plt  # for data visualization
import json
from sklearn.preprocessing import LabelEncoder

import numpy as np
import mlflow
import mlflow.sklearn
mlflow.set_tracking_uri("mlruns")
# import os
# mlflow.set_tracking_uri(os.path.join(os.path.dirname(__file__), "mlruns"))

def mlFlowVersioning(model,model_name):
    with mlflow.start_run() as run:
        # Log the model
        mlflow.sklearn.log_model(model, "model1")
        
        # Register the model
        model_uri = f"runs:/{run.info.run_id}/model1"
        mlflow.register_model(model_uri, model_name)


def train_model_partition1():
    # load dataset
    df = pd.read_csv('../data/first_data.csv') # App,Category,Installs_category,Rating,Mean_App_Sentiment,Reviews,Size_in_MB,Type,Price,Content_Rating,Genres,Last_Updated,Android_Ver
    
    
    # Encoding categorical features
    label_encoder = LabelEncoder()
    df['Category'] = label_encoder.fit_transform(df['Category'])
    df['Content_Rating'] = label_encoder.fit_transform(df['Content_Rating'])
    df['Genres'] = label_encoder.fit_transform(df['Genres'])
    df['Type'] = label_encoder.fit_transform(df['Type'])
    df['Android_Ver'] = label_encoder.fit_transform(df['Android_Ver'])
    df['Installs_category'] = label_encoder.fit_transform(df['Installs_category'])

    # choose features and label
    x = df.drop(['Installs_category', 'App', 'Last_Updated', 'Mean_App_Sentiment'], axis=1)  # Features
    y = df['Installs_category']  # Target

    # Splitting the dataset into training and other sets (80/20)
    x_train, x_other, y_train, y_other = train_test_split(x, y, test_size=0.2, random_state=28)

    # Splitting the dataset into testing and validation sets (50/50)
    x_test, x_validate, y_test, y_validate = train_test_split(x_other, y_other, test_size=0.5, random_state=28)

    # Initializing the Random Forest Regression model with 10 decision trees
    model = tree.DecisionTreeClassifier(criterion = 'entropy',max_depth= 10, min_samples_leaf= 4, min_samples_split=3)
    
    
    # Fitting the Random Forest Regression model to the data
    model.fit(x_train, y_train)

    score = model.score(x_test, y_test)
    print(score)

    y_pred = model.predict(x_test)
    #RMSE (Root Mean Square Error)
    rmse = float(format(np.sqrt(mean_squared_error(y_test, y_pred)), '.3f'))
    print("\nRMSE:\n", rmse)
    mlFlowVersioning(model,"model2-partition1")

    return model

#modell2_partition1=train_model_partition1()


def train_model_partition2():
    # load dataset
    df = pd.read_csv('../data/second_data.csv')

    # Encoding categorical features
    label_encoder = LabelEncoder()
    df['Category'] = label_encoder.fit_transform(df['Category'])
    df['Content_Rating'] = label_encoder.fit_transform(df['Content_Rating'])
    df['Genres'] = label_encoder.fit_transform(df['Genres'])
    df['Type'] = label_encoder.fit_transform(df['Type'])
    df['Android_Ver'] = label_encoder.fit_transform(df['Android_Ver'])
    df['Installs_category'] = label_encoder.fit_transform(df['Installs_category'])

    # choose features and label
    x = df.drop(['Installs_category', 'App', 'Last_Updated', 'Mean_App_Sentiment'], axis=1)  # Features
    y = df['Installs_category']  # Target

    # Splitting the dataset into training and other sets (80/20)
    x_train, x_other, y_train, y_other = train_test_split(x, y, test_size=0.2, random_state=28)

    # Splitting the dataset into testing and validation sets (50/50)
    x_test, x_validate, y_test, y_validate = train_test_split(x_other, y_other, test_size=0.5, random_state=28)

    # Initializing the Random Forest Regression model with 10 decision trees
    model = tree.DecisionTreeClassifier(criterion = 'entropy',max_depth= 6, min_samples_leaf= 2, min_samples_split=2)    
    # Fitting the Random Forest Regression model to the data
    model.fit(x_train, y_train)

    score = model.score(x_test, y_test)
    print(score)

    y_pred = model.predict(x_test)
    #RMSE (Root Mean Square Error)
    rmse = float(format(np.sqrt(mean_squared_error(y_test, y_pred)), '.3f'))
    print("\nRMSE:\n", rmse)
    
    mlFlowVersioning(model,"model2-partition2")
    return model


#modell2_partition2=train_model_partition2()

def train_model_partition3():
    # load dataset
    df = pd.read_csv('../data/finaldataset.csv')

    # Encoding categorical features
    label_encoder = LabelEncoder()
    df['Category'] = label_encoder.fit_transform(df['Category'])
    df['Content_Rating'] = label_encoder.fit_transform(df['Content_Rating'])
    df['Genres'] = label_encoder.fit_transform(df['Genres'])
    df['Type'] = label_encoder.fit_transform(df['Type'])
    df['Android_Ver'] = label_encoder.fit_transform(df['Android_Ver'])
    df['Installs_category'] = label_encoder.fit_transform(df['Installs_category'])

    # choose features and label
    x = df.drop(['Installs_category', 'App', 'Last_Updated', 'Mean_App_Sentiment'], axis=1)  # Features
    y = df['Installs_category']  # Target

    # Splitting the dataset into training and other sets (80/20)
    x_train, x_other, y_train, y_other = train_test_split(x, y, test_size=0.2, random_state=28)

    # Splitting the dataset into testing and validation sets (50/50)
    x_test, x_validate, y_test, y_validate = train_test_split(x_other, y_other, test_size=0.5, random_state=28)

    # Initializing the Random Forest Regression model with 10 decision trees
    model = tree.DecisionTreeClassifier(criterion = 'gini',max_depth=6 , min_samples_leaf=2 , min_samples_split=3)
    
    # Fitting the Random Forest Regression model to the data
    model.fit(x_train, y_train)

    score = model.score(x_test, y_test)
    print(score)

    y_pred = model.predict(x_test)
    #RMSE (Root Mean Square Error)
    rmse = float(format(np.sqrt(mean_squared_error(y_test, y_pred)), '.3f'))
    print("\nRMSE:\n", rmse)
    mlFlowVersioning(model,"model2-partition3")
    return model

#modell2_partition3=train_model_partition3()

def predict_rate(model, user_input_json):
    # Load the JSON file into a DataFrame
    #with open(user_input_json, 'r') as f:
    #    data = [json.load(f)]
    user_input = pd.DataFrame([user_input_json])

    # Encoding categorical features
    label_encoder = LabelEncoder()
    user_input.drop(['App', 'Last_Updated'], axis=1) 
    user_input['Category'] = label_encoder.fit_transform(user_input['Category'])
    user_input['Content_Rating'] = label_encoder.fit_transform(user_input['Content_Rating'])
    user_input['Genres'] = label_encoder.fit_transform(user_input['Genres'])
    user_input['Type'] = label_encoder.fit_transform(user_input['Type'])
    user_input['Android_Ver'] = label_encoder.fit_transform(user_input['Android_Ver'])
    x = user_input.drop(['App', 'Last_Updated'], axis=1)

    # Predicting the installs
    value = model.predict(x)

    # The installs category
    value = round(value[0], 0)
    if value<=0:
        result = 'no'
        installs = '0'
    elif value>0 and value <=10:
        result = 'Very low'
        installs = '0-10'
    elif value > 10 and value <=1000:
        result = 'Low'
        installs = '10-1000'
    elif value >1000 and value <= 10000:
        result = 'Moderate'
        installs = '1000-10000'
    elif value >10000 and value <= 100000:
        result = 'More than Moderate'
        installs = '1000-100000'
    elif value >100000 and value <= 1000000:
        result = 'High'
        installs = '100000-1000000'
    elif value >1000000 and value <= 10000000:
        result = 'Very High'
        installs = '1000000-10000000'
    else:
        result = 'Top Notch'
        installs = '10000000+'

    
    # Save result in json format   
    data = {
        "Installs_category": result,
        "Installs": installs
    }
    #file_name = 'data.json'
    #with open(file_name, 'w') as json_file:
    #    json.dump(data, json_file, indent=4)
    return data