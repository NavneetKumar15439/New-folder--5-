import joblib
from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
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

train_df_raw = pd.read_csv('train.csv')
test_df_raw = pd.read_csv('test.csv')

y_train = train_df_raw['Transported'].map({True: 1, False: 0})
train_df_raw.drop('Transported', axis=1, inplace=True)

full_df = pd.concat([train_df_raw, test_df_raw], ignore_index=True)

full_df_processed = prepare_data(full_df.copy())

group_sizes = full_df_processed.groupby('Group')['Group'].transform('count')
full_df_processed['GroupSize'] = group_sizes

processed_train_for_plot = full_df_processed.iloc[:len(train_df_raw)].copy()
processed_train_for_plot['Transported'] = y_train

plt.figure(figsize=(6, 4))
sns.countplot(x='Transported', data=processed_train_for_plot)
plt.title('Distribution of Target Variable (Transported)')
plt.xticks([0, 1], ['Not Transported (0)', 'Transported (1)'])
plt.savefig('target_distribution.png')

plt.figure(figsize=(8, 5))
sns.barplot(x='HomePlanet', y='Transported', data=processed_train_for_plot) 
plt.title('Transported Rate by HomePlanet')
plt.ylabel('Transported Rate (Mean)')
plt.savefig('homeplanet_analysis.png')

plt.figure(figsize=(10, 6))
sns.histplot(data=processed_train_for_plot, x='Age', hue='Transported', kde=True, bins=30)
plt.title('Age Distribution by Transported Status')
plt.savefig('age_distribution_analysis.png')

plt.figure(figsize=(10, 6))
sns.boxplot(x='Transported', y=np.log1p(processed_train_for_plot['TotalSpent']), data=processed_train_for_plot)
plt.title('Log(Total Spent) Distribution by Transported Status')
plt.xticks([0, 1], ['Not Transported(0)', 'Transported(1)'])
plt.ylabel('Log(Total Spent)')
plt.savefig('total_spent_analysis.png')


full_df_processed.drop(['PassengerId', 'Name', 'Cabin', 'CabinNum', 'Group'], 
                       axis=1, errors='ignore', inplace=True) # <-- ADDED 'Cabin' HERE

full_df_processed = pd.get_dummies(
    full_df_processed, columns=['HomePlanet', 'Destination', 'Deck', 'Side'], drop_first=True
)

X_train = full_df_processed.iloc[:len(train_df_raw)].copy()
X_test_final = full_df_processed.iloc[len(train_df_raw):].copy()

numerical_features = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'TotalSpent']
scaler = StandardScaler()

X_train.loc[:, numerical_features] = scaler.fit_transform(X_train[numerical_features])
X_test_final.loc[:, numerical_features] = scaler.transform(X_test_final[numerical_features])

joblib.dump(scaler, 'scaler.pkl')

print("Final X_train shape:", X_train.shape)
print("Final X_test_final shape:", X_test_final.shape)

print("Training Logistic Regression Model...")
logreg_model = LogisticRegression(solver='liblinear', random_state=42)
logreg_model.fit(X_train, y_train)

print("Model Training Complete")

joblib.dump(logreg_model, 'logreg_model.pkl')
print("Trained model saved as 'logreg_model.pkl'.")