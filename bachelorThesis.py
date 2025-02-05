######################################################
## WRITE YOUR CREDENTIALS FOR SAFECTORY TO RUN CODE ##
######################################################
email = ""        
password = ""     
selected_model = "Random Forest"
#selected_model = "Neural Network"     
#selected_model = "Gradient Boosting"                          
######################################################

import requests
from datetime import datetime
import pandas as pd
import math
import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBClassifier

cattleData = {}
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

def createSession():
    return requests.Session()

def login(session):
    login_url = "https://devtrack.safectory.com/api/session"
    login_data = {
        "email": email,
        "password": password
    }
    response = session.post(login_url, data=login_data)

    # Return the response for further processing
    return response

def proceedData():
    session = createSession()
    loginResponse = login(session)

    if loginResponse.status_code == 200:
        print("Login successful!")
        
        protected_url = "https://devtrack.safectory.com/api/devices"
        api_response = session.get(protected_url)
        
        if api_response.status_code == 200:
            print("Access to protected resource successful!")
            print(api_response.json()[0]["id"])
        
        url = "https://devtrack.safectory.com/api/reports/summary"
        params = {
            "deviceId": "177657162",
            #"deviceId": api_response.json()[0]["id"],
            "from": "2024-08-22T00:00:00.000Z",
            "to": "2024-08-22T02:00:00.000Z"
        }

        # Send the GET request with the parameters using the same session
        response = session.get(url, params=params)

        # Check if the request was successful
        if response.status_code == 200:
            print("Request successful!")
            print(response.json())
        else:
            print(f"Request failed with status code {response.status_code}")
            print(response.text)
        

        url = "https://devtrack.safectory.com/api/reports/route"

        # Define the parameters for the request
        params = {
            "deviceId": "177657162",  # List of device IDs or [0] for all devices
            "from": "2024-11-21T00:00:00.000Z",  # Start time in UTC format
            "to": "2024-11-24T00:00:00.000Z",    # End time in UTC format
            "filterSameLocation": False,
            "filterFlapping": False,
            "filterNonBeacon": False,
            "useServerTime": False,
            "includeNoRssiFilterDevices": False
        }

        # Headers
        headers = {
            "Accept": "application/json"
        }

        # Send the GET request with the parameters and headers
        response = session.get(url, headers=headers, params=params)

        # Check if the request was successful
        if response.status_code == 200:
            print("Request successful!")
            sorted_data = sorted(response.json(), key=lambda x: datetime.fromisoformat(x['serverTime'].replace('Z', '+00:00')))
            for x in sorted_data:
                #print(x["rssi"], x["serverTime"], x["beaconName"])
                # Ensure 'serverTime' exists and 'rssi' is not None
                if x["rssi"] is not None and x["serverTime"] not in cattleData:
                    # Store the 'rssi' and 'beaconName' under the 'serverTime' key
                    cattleData[x["serverTime"]] = {"rssi": x["rssi"], "beaconName": x["beaconName"]}
                elif x["rssi"] is not None and cattleData[x["serverTime"]]["rssi"] < x["rssi"]:
                    # Update 'rssi' and 'beaconName' if new 'rssi' is greater
                    cattleData[x["serverTime"]]["rssi"] = x["rssi"]
                    cattleData[x["serverTime"]]["beaconName"] = x["beaconName"]
        else:
            print(f"Request failed with status code {response.status_code}")
            print(response.text)  # Print the response for debugging purposes

    else:
        print(f"Login failed: {loginResponse.status_code}")
        print(loginResponse.text)

    df = pd.DataFrame(response.json())


    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    # Function to format the serverTime
    def format_server_time(server_time):
        # Parse the server time string and remove milliseconds and timezone
        return datetime.fromisoformat(server_time.replace('Z', '+00:00')).strftime('%Y-%m-%d %H:%M:%S')

    # Convert cattleData to a list of dictionaries, formatting the serverTime
    data = [{"serverTime": format_server_time(time), "beaconName": cattleData[time]["beaconName"]} for time in cattleData]

    # Convert to a pandas DataFrame
    df = pd.DataFrame(data)

    df = pd.DataFrame(data)
    df['serverTime'] = pd.to_datetime(df['serverTime'])

    # Create the 'end_time' column by shifting the 'serverTime' up by one row
    df['end_time'] = df['serverTime'].shift(-1)
    df['beaconName'] = df['beaconName'].shift(-1)

    # Create a new DataFrame with the necessary columns
    summary_df = df[['serverTime', 'end_time', 'beaconName']].copy()
    summary_df.columns = ['start_time', 'end_time', 'beacon']

    # Optionally, remove the last row if 'end_time' is NaN
    summary_df = summary_df.dropna(subset=['end_time'])

    # Detect changes in the beacon to create a group identifier
    summary_df['group'] = (summary_df['beacon'] != summary_df['beacon'].shift()).cumsum()

    # Group by the created 'group' identifier to combine consecutive entries
    merged_df = summary_df.groupby('group').agg(
        start_time=('start_time', 'first'),
        end_time=('end_time', 'last'),
        beacon=('beacon', 'first')  # Ensure the beacon name is retained after grouping
    ).reset_index(drop=True)

    # Calculate the time difference in seconds
    merged_df['duration'] = (merged_df['end_time'] - merged_df['start_time']).dt.total_seconds()

    # Define behaviors with a single apply
    merged_df['behavior'] = merged_df.apply(
        lambda row: 'Drinking' if 'Tränke' in row['beacon'] and row['duration'] >= 61 else (
            'Milking' if 'AMS' in row['beacon'] and row['duration'] >= 60 else ''
        ),
        axis=1
    )

    protected_url = "https://devtrack.safectory.com/api/beacons"
    api_response = session.get(protected_url)

    if api_response.status_code == 200:
        print("Access to protected resource successful!")
        
        # Parse the JSON response
        json_data = api_response.json()

        desired_names = [
            "W10T_A7H_B-1-17", 
            "W10T_A7H_B-1-18 Tränke", 
            "W10T_A7H_B-1-16",
            "SF:00:00:00:00:00:DE",
            "W10T_A7H_B-1-15 Bürste",
            "SF:00:00:00:00:00:E1",
            "W10T_A7H_B-1-14",
            "W10T_A7H_B-1-11",
            "W10T_A7H_B-1-13",
            "W10T_A7H_B-1-12",
            "SF:00:00:00:00:00:DF",
            "SF:00:00:00:00:00:DC",
            "W10T_A7H_B-1-8 AMS",
            "W10T_A7H_B-1-19 Tränke",
            "W10T_A7H_B-1-7",
            "W10T_A7H_B-1-6",
            "W10T_A7H_B-1-5",
            "W10T_A7H_B-1-1",
            "W10T_A7H_B-1-2",
            "W10T_A7H_B-1-3",
            "W10T_A7H_B-1-4",
        ]

        # Filter the data based on the name field
        filtered_data = [
            {"name": entry["name"], "latitude": entry["latitude"], "longitude": entry["longitude"]}
            for entry in json_data if entry.get("name") in desired_names
        ]

        # Function to calculate the Haversine distance
        def haversine(lat1, lon1, lat2, lon2):
            R = 6371  # Radius of Earth in kilometers
            lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
            dlat = lat2 - lat1
            dlon = lon2 - lon1
            a = math.sin(dlat / 2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2)**2
            c = 2 * math.asin(math.sqrt(a))
            return R * c

        # Calculate distances between all filtered beacons
        distances = []
        for i, beacon1 in enumerate(filtered_data):
            for j, beacon2 in enumerate(filtered_data):
                if i != j:  # Avoid calculating distance to itself
                    distance = haversine(
                        beacon1["latitude"], beacon1["longitude"],
                        beacon2["latitude"], beacon2["longitude"]
                    )
                    distances.append({
                        "from": beacon1["name"],
                        "to": beacon2["name"],
                        "distance_km": round(distance, 3)  # Round to 3 decimal places
                    })

        # Convert the results to JSON
        output_json_str = json.dumps(distances, indent=4, ensure_ascii=False)

    # Parse the JSON string to a Python object
    output_json = json.loads(output_json_str)

    # Convert distance to meters
    for item in output_json:
        item["distance_m"] = item["distance_km"] * 1000

    # Threshold: Meters per second
    walking_threshold = 0.84  # 8.4 meters in 10 seconds

    # Add walking behavior to merged_df
    def check_walking_behavior(row, next_row, distances):
        if next_row is None:  # Skip the last row as there's no next beacon
            return ""
        
        # Get current and next beacons
        current_beacon = row["beacon"]
        next_beacon = next_row["beacon"]
        duration = next_row["duration"]

        next_behaviour = next_row["behavior"]

        if next_behaviour == "Drinking":
            return "Drinking"

        if next_behaviour == "Milking":
            return "Milking"
        
        # Match beacons in distances
        for item in distances:
            if ((item["from"] == current_beacon and item["to"] == next_beacon) or
                (item["from"] == next_beacon and item["to"] == current_beacon)):
                distance = item["distance_m"]
                # Check if distance/duration matches the walking threshold
                if distance / duration >= walking_threshold:
                    return "Walking"
        return "Resting"

    # Apply the behavior calculation
    for i in range(len(merged_df) - 1):
        merged_df.loc[i + 1, "behavior"] = check_walking_behavior(
            merged_df.loc[i],
            merged_df.loc[i + 1],
            output_json
        )
    return merged_df

def runRandomForestModel(merged_df):
    # Example DataFrame (replace this with your merged_df)
    train_df = merged_df  # Your actual DataFrame

    # Convert categorical column 'beacon' to dummy variables
    train_df = pd.get_dummies(train_df, columns=["beacon"])

    # Add other features like time-based features if not already present
    train_df["hour"] = pd.to_datetime(train_df["start_time"]).dt.hour
    train_df["day"] = pd.to_datetime(train_df["start_time"]).dt.dayofweek

    # Define features and target
    features = [col for col in train_df.columns if col.startswith('beacon_') or col in ["hour", "day", "duration"]]
    target = "behavior"

    # Split the data
    train_df, test_df = train_test_split(train_df, test_size=0.2, shuffle=False)

    # Train the Random Forest model
    model = RandomForestClassifier()
    model.fit(train_df[features], train_df[target])

    # Evaluate the model
    predictions = model.predict(test_df[features])

    # Calculate metrics
    accuracy = accuracy_score(test_df[target], predictions)
    precision = precision_score(test_df[target], predictions, average='weighted')  # Weighted for multi-class
    recall = recall_score(test_df[target], predictions, average='weighted')
    f1 = f1_score(test_df[target], predictions, average='weighted')

    # Print metrics
    print(f"Accuracy: {accuracy:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1 Score: {f1:.2f}")

    # Generate a detailed classification report
    print("\nClassification Report:")
    print(classification_report(test_df[target], predictions))

    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import confusion_matrix

    # Compute the confusion matrix
    conf_matrix = confusion_matrix(test_df[target], predictions)

    # Plot the confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=train_df[target].unique(), yticklabels=train_df[target].unique())
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()

    # Get feature importance from the Random Forest model
    importances = model.feature_importances_
    feature_names = features

    # Plot the feature importance
    plt.figure(figsize=(10, 6))
    sns.barplot(x=importances, y=feature_names, palette="viridis")
    plt.title("Feature Importance")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.show()
    


def runNeuralNetworkModel(merged_df):
    # Load your DataFrame (replace 'merged_df' with your actual DataFrame)
    df = merged_df.copy()

    # Encode the target variable (behavior)
    label_encoder = LabelEncoder()
    df['behavior_encoded'] = label_encoder.fit_transform(df['behavior'])

    # One-hot encode the 'beacon' feature
    df = pd.get_dummies(df, columns=['beacon'], drop_first=True)

    # Add time-based features if applicable
    df['hour'] = pd.to_datetime(df['start_time']).dt.hour
    df['day'] = pd.to_datetime(df['start_time']).dt.dayofweek

    # Define features and target
    features = [col for col in df.columns if col not in ['start_time', 'end_time', 'behavior', 'behavior_encoded']]
    X = df[features]
    y = df['behavior_encoded']

    # Normalize numerical features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # One-hot encode the target variable
    y = to_categorical(y)

    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize the ANN
    model = Sequential()

    # Input layer
    model.add(Dense(units=64, activation='relu', input_dim=X_train.shape[1]))

    # Hidden layers
    model.add(Dense(units=128, activation='relu'))
    model.add(Dropout(0.3))  # Dropout for regularization
    model.add(Dense(units=64, activation='relu'))

    # Output layer (number of classes in 'behavior')
    model.add(Dense(units=y_train.shape[1], activation='softmax'))

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=50, batch_size=32, verbose=1)

    # Evaluate the model on the test set
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Accuracy: {accuracy:.2f}")

    # Plot training history
    # Accuracy plot
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

    # Loss plot
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    # Predict on the test set
    y_pred = model.predict(X_test)

    # Convert predictions back to labels
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true = np.argmax(y_test, axis=1)

    from sklearn.metrics import classification_report
    import numpy as np

    # Ensure target_names match the unique classes in y_true
    unique_classes = np.unique(y_true)  # Get unique classes from the test data
    class_labels = label_encoder.inverse_transform(unique_classes)  # Map back to original labels

    # Generate and print the classification report
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred_classes, target_names=class_labels))
    
def runGradientBoostingModel(merged_df):

    # Load and preprocess the data
    df = merged_df.copy()

    # Encode the target variable (behavior)
    label_encoder = LabelEncoder()
    df['behavior_encoded'] = label_encoder.fit_transform(df['behavior'])

    # One-hot encode the 'beacon' feature
    df = pd.get_dummies(df, columns=['beacon'], drop_first=True)

    # Add time-based features
    df['hour'] = pd.to_datetime(df['start_time']).dt.hour
    df['day'] = pd.to_datetime(df['start_time']).dt.dayofweek

    # Define features and target
    features = [col for col in df.columns if col not in ['start_time', 'end_time', 'behavior', 'behavior_encoded']]
    X = df[features]
    y = df['behavior_encoded']

    # Normalize numerical features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and train the XGBoost model
    xgb_model = XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        use_label_encoder=False,
        eval_metric='mlogloss'
    )
    xgb_model.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = xgb_model.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy: {accuracy:.2f}")

    from sklearn.metrics import classification_report

    # Get the classes in y_test
    labels_in_test = sorted(set(y_test))

    # Generate classification report with only the labels present in y_test
    print("\nClassification Report:")
    print(classification_report(
        y_test,
        y_pred,
        target_names=[label_encoder.classes_[label] for label in labels_in_test],
        labels=labels_in_test
    ))

    from sklearn.metrics import roc_curve, auc
    from sklearn.preprocessing import label_binarize
    import matplotlib.pyplot as plt

    # Binarize the output for ROC curve
    y_test_binarized = label_binarize(y_test, classes=range(len(label_encoder.classes_)))

    # Predict probabilities
    y_pred_proba = xgb_model.predict_proba(X_test)

    # Plot ROC Curve for each class
    plt.figure(figsize=(10, 8))
    for i, class_name in enumerate(label_encoder.classes_):
        fpr, tpr, _ = roc_curve(y_test_binarized[:, i], y_pred_proba[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"Class {class_name} (AUC = {roc_auc:.2f})")

    plt.plot([0, 1], [0, 1], 'k--')  # Random guess line
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve for XGBoost Model")
    plt.legend(loc="lower right")
    plt.show()

    from sklearn.metrics import precision_recall_curve

    # Plot Precision-Recall curve
    plt.figure(figsize=(10, 8))
    for i, class_name in enumerate(label_encoder.classes_):
        precision, recall, _ = precision_recall_curve(y_test_binarized[:, i], y_pred_proba[:, i])
        plt.plot(recall, precision, label=f"Class {class_name}")

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve for XGBoost Model")
    plt.legend(loc="lower left")
    plt.show()

if __name__ == "__main__":
    dataframe = proceedData()
    if selected_model == "Random Forest":
        runRandomForestModel(dataframe)
    elif selected_model == "Neural Network":
        runNeuralNetworkModel(dataframe)
    elif selected_model == "Gradient Boosting":
        runGradientBoostingModel(dataframe)
