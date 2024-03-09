import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
import pickle

# Read the dataset
df = pd.read_csv('laptop_data.csv')

# Drop the 'Unnamed: 0' column
df.drop(columns=['Unnamed: 0'], inplace=True)

# Preprocessing steps for Ram and Weight columns
df['Ram'] = df['Ram'].str.replace('GB', '').astype('int32')
df['Weight'] = df['Weight'].str.replace('kg', '').astype('float32')

# Create Ips column
df['Ips'] = df['ScreenResolution'].apply(lambda x: 1 if 'IPS' in x else 0)

# Split ScreenResolution column into X_res and Y_res
new = df['ScreenResolution'].str.split('x', n=1, expand=True)
df['X_res'] = new[0].str.replace(',', '').str.findall(r'(\d+\.?\d+)').apply(lambda x: x[0]).astype('int')
df['Y_res'] = new[1].astype('int')

# Calculate ppi and drop ScreenResolution column
df['ppi'] = (((df['X_res'] ** 2) + (df['Y_res'] ** 2)) ** 0.5 / df['Inches']).astype('float')
df.drop(columns=['ScreenResolution', 'Inches', 'X_res', 'Y_res'], inplace=True)

# Extract processor name and brand
df['Cpu Name'] = df['Cpu'].apply(lambda x: " ".join(x.split()[0:3]))
df['Cpu brand'] = df['Cpu Name'].apply(lambda x: x if x in ['Intel Core i7', 'Intel Core i5', 'Intel Core i3'] else 'Other Intel Processor' if x.startswith('Intel') else 'AMD Processor')
df.drop(columns=['Cpu', 'Cpu Name'], inplace=True)

# Preprocessing steps for Memory column
df['Memory'] = df['Memory'].astype(str).replace('\.0', '', regex=True)
df["Memory"] = df["Memory"].str.replace('GB', '').str.replace('TB', '000')
new = df["Memory"].str.split("+", n=1, expand=True)
df["first"] = new[0].str.extract(r'(\d+)').astype(float)
df["second"] = new[1].str.extract(r'(\d+)').fillna("0").astype(int)

# Create HDD and SSD columns
df["Layer1HDD"] = df["first"].apply(lambda x: 1 if "HDD" in df['Memory'].values else 0)
df["Layer1SSD"] = df["first"].apply(lambda x: 1 if "SSD" in df['Memory'].values else 0)
df["Layer2HDD"] = df["second"].apply(lambda x: 1 if "HDD" in df['Memory'].values else 0)
df["Layer2SSD"] = df["second"].apply(lambda x: 1 if "SSD" in df['Memory'].values else 0)

# Calculate HDD and SSD sizes
df["HDD"] = (df["first"] * df["Layer1HDD"] + df["second"] * df["Layer2HDD"])
df["SSD"] = (df["first"] * df["Layer1SSD"] + df["second"] * df["Layer2SSD"])

# Drop unnecessary columns
df.drop(columns=['Memory', 'first', 'second', 'Layer1HDD', 'Layer1SSD', 'Layer2HDD', 'Layer2SSD'], inplace=True)

# Extract GPU brand
df['Gpu brand'] = df['Gpu'].apply(lambda x: x.split()[0])

# Filter out rows with ARM GPU
df = df[df['Gpu brand'] != 'ARM']
df.drop(columns=['Gpu'], inplace=True)

# Categorize OS
def cat_os(inp):
    if inp in ['Windows 10', 'Windows 7', 'Windows 10 S']:
        return 'Windows'
    elif inp in ['macOS', 'Mac OS X']:
        return 'Mac'
    else:
        return 'Others/No OS/Linux'

df['os'] = df['OpSys'].apply(cat_os)
df.drop(columns=['OpSys'], inplace=True)

# Drop unnecessary columns
df.drop(columns=['TypeName', 'Ips', 'ppi', 'Gpu brand'], inplace=True)

# Split data into features (X) and target (y)
X = df.drop(columns=['Price'])
y = np.log(df['Price'])

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=2)

# Define preprocessing steps and model
step1 = ColumnTransformer(transformers=[
    ('col_tnf', OneHotEncoder(sparse_output=False, drop='first'), [0, 3, 6])
], remainder='passthrough')

step2 = RandomForestRegressor(n_estimators=100,
                              random_state=2,
                              max_samples=0.5,
                              max_features=0.75,
                              max_depth=15)

# Define the pipeline
pipe = Pipeline([
    ('step1', step1),
    ('step2', step2)
])

# Fit the pipeline on the training data
pipe.fit(X_train, y_train)

# Predict on the test data
y_pred = pipe.predict(X_test)

# Print evaluation metrics
print('R2 score:', r2_score(y_test, y_pred))
print('MAE:', mean_absolute_error(y_test, y_pred))

# Define the columns for user input
columns = ['Company', 'Ram', 'Weight', 'Cpu brand', 'HDD', 'SSD', 'os']

# Get user input
user_input = []
for column in columns:
    value = input(f"Enter {column}: ")
    user_input.append(value)

# Create a DataFrame from user input
user_input_df = pd.DataFrame([user_input], columns=columns)

# Predict the price for user input
y_pred = pipe.predict(user_input_df)
log_price = y_pred

# Convert back to the normal amount
normal_price = np.exp(log_price)
print('Predicted Price:', normal_price)

# Save the model
with open('model.pkl', 'wb') as file:
    pickle.dump(pipe, file)
