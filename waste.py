import numpy as np
import pandas as pd
from flask import Flask, render_template, request
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor

# Read the CSV file into a DataFrame
df = pd.read_csv("laptop_data.csv")

# Drop unnecessary columns
df.drop(columns=['Unnamed: 0'], inplace=True)

# Preprocess 'Ram' and 'Weight' columns
df['Ram'] = df['Ram'].str.replace('GB','').astype('int32')
df['Weight'] = df['Weight'].str.replace('kg','').astype('float32')

# Create 'Ips' column
df['Ips'] = df['ScreenResolution'].apply(lambda x:1 if 'IPS' in x else 0)

# Split 'ScreenResolution' into 'X_res' and 'Y_res' columns
new = df['ScreenResolution'].str.split('x',n=1,expand=True)
df['X_res'] = new[0].str.replace(',','').str.findall(r'(\d+\.?\d+)').apply(lambda x:x[0]).astype('int')
df['Y_res'] = new[1].astype('int')

# Calculate 'ppi'
df['ppi'] = (((df['X_res']**2) + (df['Y_res']**2))**0.5/df['Inches']).astype('float')

# Drop unnecessary columns
df.drop(columns=['ScreenResolution', 'Inches', 'X_res', 'Y_res', 'Layer1Hybrid', 'Layer1Flash_Storage',
                 'Layer2Hybrid', 'Layer2Flash_Storage', 'Memory', 'Hybrid', 'Flash_Storage', 'Gpu',
                 'TypeName', 'Ips', 'ppi', 'Gpu brand', 'OpSys'], inplace=True)

# Create 'Cpu brand' column
df['Cpu Name'] = df['Cpu'].apply(lambda x:" ".join(x.split()[0:3]))
df['Cpu brand'] = df['Cpu Name'].apply(lambda x: x if x in ['Intel Core i7', 'Intel Core i5', 'Intel Core i3'] else ('Other Intel Processor' if x.split()[0] == 'Intel' else 'AMD Processor'))
df.drop(columns=['Cpu', 'Cpu Name'], inplace=True)

# Parse 'Memory' column
df['Memory'] = df['Memory'].astype(str).replace('\.0', '', regex=True)
df["Memory"] = df["Memory"].str.replace('GB', '')
df["Memory"] = df["Memory"].str.replace('TB', '000')
new = df["Memory"].str.split("+", n = 1, expand = True)
df["first"]= new[0].str.strip()
df["second"]= new[1]
df["first"] = df['first'].str.replace(r'\D', '').astype(int)
df["second"].fillna("0", inplace = True)
df["second"] = df['second'].str.replace(r'\D', '').astype(int)
df["HDD"]=(df["first"]*df["Layer1HDD"]+df["second"]*df["Layer2HDD"])
df["SSD"]=(df["first"]*df["Layer1SSD"]+df["second"]*df["Layer2SSD"])
df.drop(columns=['first', 'second', 'Layer1HDD', 'Layer1SSD', 'Layer2HDD', 'Layer2SSD'], inplace=True)

# Map 'os' values
df['os'] = df['OpSys'].apply(lambda x: 'Windows' if x in ['Windows 10', 'Windows 7', 'Windows 10 S'] else ('Mac' if x in ['macOS', 'Mac OS X'] else 'Others/No OS/Linux'))
df.drop(columns=['OpSys'], inplace=True)

# Create and train model pipeline
X = df.drop(columns=['Price'])
y = np.log(df['Price'])

step1 = ColumnTransformer(transformers=[('col_tnf',OneHotEncoder(sparse=False,drop='first'),[0, 3, 6])], remainder='passthrough')
step2 = RandomForestRegressor(n_estimators=100, random_state=2, max_samples=0.5, max_features=0.75, max_depth=15)
pipe = Pipeline([('step1',step1), ('step2',step2)])
pipe.fit(X, y)

# Define Flask app
app = Flask(__name__)

# Define column names
columns = ['Company', 'Ram', 'Weight', 'Cpu brand', 'HDD', 'SSD', 'os']

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        user_input = [request.form[col] for col in columns]
        user_input_df = pd.DataFrame([user_input], columns=columns)
        y_pred = pipe.predict(user_input_df)
        normal_price = np.exp(y_pred)
        return render_template('result.html', prediction=normal_price[0])
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
