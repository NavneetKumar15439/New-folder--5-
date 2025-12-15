import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

def prepare_data(df):
    """Applies preprocessing and feature engineering."""

    df[['Deck', 'CabinNum', 'Side']] = df['Cabin'].str.split('/', expand=True)
    df['Group'] = df['PassengerId'].str.split('_', expand=True)[0].astype(str)

    spending_cols = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
    df['TotalSpent'] = df[spending_cols].sum(axis=1)

    numerical_features = ['Age'] + spending_cols + ['TotalSpent']
    categorical_features = ['HomePlanet', 'CryoSleep', 'Destination', 'VIP', 'Deck', 'Side']

    for col in numerical_features:
        df.loc[:, col] = df[col].fillna(df[col].median())

    for col in categorical_features:
        df.loc[:, col] = df[col].fillna(df[col].mode()[0])

    for col in ['CryoSleep', 'VIP']:
        df.loc[:, col] = df[col].astype(int)
        
    return df


test_df_raw = pd.read_csv('test.csv')
passenger_ids = test_df_raw['PassengerId']

test_df_processed = test_df_raw.copy()
X_test = prepare_data(test_df_processed)

group_sizes = X_test.groupby('Group')['Group'].transform('count')
X_test['GroupSize'] = group_sizes

X_test.drop(['PassengerId', 'Name', 'Cabin', 'CabinNum', 'Group'], 
            axis=1, errors='ignore', inplace=True) # <-- ADDED 'Cabin' HERE

X_test = pd.get_dummies(
    X_test, columns=['HomePlanet', 'Destination', 'Deck', 'Side'], drop_first=True
)

try:
    scaler = joblib.load('scaler.pkl')
    logreg_model = joblib.load('logreg_model.pkl')
except FileNotFoundError:
    print("Error: Model or Scaler not found. Run train.py first to create 'scaler.pkl' and 'logreg_model.pkl'.")
    exit()

numerical_features = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'TotalSpent']
X_test.loc[:, numerical_features] = scaler.transform(X_test[numerical_features])

print("Making predictions on test data...")
predictions_numeric = logreg_model.predict(X_test)
prediction_bool = (predictions_numeric == 1)

submission_df = pd.DataFrame({
    'PassengerId': passenger_ids,
    'Transported': prediction_bool
})

submission_file_name = 'submission_predictions.xlsx'
submission_df.to_excel(submission_file_name, index=False, engine='openpyxl')

print(f"Prediction complete. Submission saved as '{submission_file_name}'.")