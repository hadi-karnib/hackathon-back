# Importing the libraries
from sklearn.ensemble import RandomForestRegressor  # for building the model
# for calculating the cost function
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split  # for splitting the data
import numpy as np  # for array operations
import pandas as pd  # for working with DataFrames
import matplotlib.pyplot as plt  # for data visualization
from sklearn import tree

from sklearn.preprocessing import LabelEncoder
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn import tree
from sklearn.metrics import mean_squared_error
import numpy as np
import mlflow
import mlflow.sklearn

mlflow.set_tracking_uri("mlruns")

def mlFlowVersioning(class_tree, model_name):
    with mlflow.start_run() as run:
        # Log the model
        mlflow.sklearn.log_model(class_tree, "model1")
        
        # Register the model
        model_uri = f"runs:/{run.info.run_id}/model1"
        mlflow.register_model(model_uri, model_name)

def train_model_partition1():
    # load dataset
    df = pd.read_csv('../data/first_data.csv') # app, category, size, type, price, content rating, genre last updated, android version
    label_encoder = LabelEncoder()
    df['Category'] = label_encoder.fit_transform(df['Category'])
    df['Content_Rating'] = label_encoder.fit_transform(df['Content_Rating'])
    df['Genres'] = label_encoder.fit_transform(df['Genres'])
    df['Type'] = label_encoder.fit_transform(df['Type'])
    df['Android_Ver'] = label_encoder.fit_transform(df['Android_Ver'])
    df['Installs_category'] = label_encoder.fit_transform(df['Installs_category'])
    # choose features and label
    x = df.drop(['Installs_category', 'App', 'Last_Updated', 'Rating', 'Mean_App_Sentiment', 'Reviews'], axis=1)  # Features
    y = df['Installs_category']  # Target

    # Splitting the dataset into training and other sets (80/20)
    x_train, x_other, y_train, y_other = train_test_split(x, y, test_size=0.2, random_state=28)

    # Splitting the dataset into testing and validation sets (50/50)
    x_test, x_validate, y_test, y_validate = train_test_split(x_other, y_other, test_size=0.5, random_state=28)

    class_tree = tree.DecisionTreeClassifier(criterion= 'entropy', max_depth= 8, min_samples_leaf= 2, min_samples_split= 2)
    class_tree.fit(x_train, y_train)

    y_pred = class_tree.predict(x_test)

    score = class_tree.score(x_test, y_test)
    print(score)

    rmse = float(format(np.sqrt(mean_squared_error(y_test, y_pred)), '.3f'))
    print("\nRMSE:\n", rmse)

    # Start an MLflow run
    mlFlowVersioning(class_tree,"model1-partition1")
    return class_tree

#modell1_partition1=train_model_partition1()

def train_model_partition2():
    # load dataset
    df = pd.read_csv('../data/second_data.csv') # app, category, size, type, price, content rating, genre last updated, android version
    label_encoder = LabelEncoder()
    df['Category'] = label_encoder.fit_transform(df['Category'])
    df['Content_Rating'] = label_encoder.fit_transform(df['Content_Rating'])
    df['Genres'] = label_encoder.fit_transform(df['Genres'])
    df['Type'] = label_encoder.fit_transform(df['Type'])
    df['Android_Ver'] = label_encoder.fit_transform(df['Android_Ver'])
    df['Installs_category'] = label_encoder.fit_transform(df['Installs_category'])
    # choose features and label
    x = df.drop(['Installs_category', 'App', 'Last_Updated', 'Rating', 'Mean_App_Sentiment', 'Reviews'], axis=1)  # Features
    y = df['Installs_category']  # Target

    # Splitting the dataset into training and other sets (80/20)
    x_train, x_other, y_train, y_other = train_test_split(x, y, test_size=0.2, random_state=28)

    # Splitting the dataset into testing and validation sets (50/50)
    x_test, x_validate, y_test, y_validate = train_test_split(x_other, y_other, test_size=0.5, random_state=28)

    class_tree = tree.DecisionTreeClassifier(criterion= 'gini', max_depth= 10, min_samples_leaf= 4, min_samples_split= 10)
    class_tree.fit(x_train, y_train)

    y_pred = class_tree.predict(x_test)

    score = class_tree.score(x_test, y_test)
    print(score)

    rmse = float(format(np.sqrt(mean_squared_error(y_test, y_pred)), '.3f'))
    print("\nRMSE:\n", rmse)
    # Start an MLflow run
    mlFlowVersioning(class_tree,"model1-partition2")

    return class_tree

#modell1_partition2=train_model_partition2()


def train_model_partition3():
    # load dataset
    df = pd.read_csv('../data/finaldataset.csv') # app, category, size, type, price, content rating, genre last updated, android version
    label_encoder = LabelEncoder()
    df['Category'] = label_encoder.fit_transform(df['Category'])
    df['Content_Rating'] = label_encoder.fit_transform(df['Content_Rating'])
    df['Genres'] = label_encoder.fit_transform(df['Genres'])
    df['Type'] = label_encoder.fit_transform(df['Type'])
    df['Android_Ver'] = label_encoder.fit_transform(df['Android_Ver'])
    df['Installs_category'] = label_encoder.fit_transform(df['Installs_category'])
    
    # choose features and label
    x = df.drop(['Installs_category', 'App', 'Last_Updated', 'Rating', 'Mean_App_Sentiment', 'Reviews'], axis=1)  # Features
    y = df['Installs_category']  # Target

    # Splitting the dataset into training and other sets (80/20)
    x_train, x_other, y_train, y_other = train_test_split(x, y, test_size=0.2, random_state=28)

    # Splitting the dataset into testing and validation sets (50/50)
    x_test, x_validate, y_test, y_validate = train_test_split(x_other, y_other, test_size=0.5, random_state=28)

    class_tree = tree.DecisionTreeClassifier(criterion='gini', max_depth=10, min_samples_leaf=2, min_samples_split=5)
    class_tree.fit(x_train, y_train)

    y_pred = class_tree.predict(x_test)

    score = class_tree.score(x_test, y_test)
    print(score)

    rmse = float(format(np.sqrt(mean_squared_error(y_test, y_pred)), '.3f'))
    print("\nRMSE:\n", rmse)

    # Start an MLflow run
    mlFlowVersioning(class_tree,"model1-partition3")

    return class_tree



#modell1_partition3=train_model_partition3()


def predict_rate(model, user_input_json):
    # Load the JSON file into a DataFrame
    #with open(user_input_json, 'r') as f:
    #    data = [json.load(f)]
    user_input = pd.DataFrame([user_input_json])

    label_encoder = LabelEncoder()
    user_input.drop(['App', 'Last_Updated'], axis=1) 
    user_input['Category'] = label_encoder.fit_transform(user_input['Category'])
    user_input['Content_Rating'] = label_encoder.fit_transform(user_input['Content_Rating'])
    user_input['Genres'] = label_encoder.fit_transform(user_input['Genres'])
    user_input['Type'] = label_encoder.fit_transform(user_input['Type'])
    user_input['Android_Ver'] = label_encoder.fit_transform(user_input['Android_Ver'])
    x = user_input.drop(['App', 'Last_Updated'], axis=1)
    value = model.predict(x)
    value = round(value[0], 0)
    # The installs category
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
    
    # Save the data to a JSON file
    #with open(file_name, 'w') as json_file:
    #    json.dump(data, json_file, indent=4)
    return data