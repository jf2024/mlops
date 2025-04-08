import pandas as pd
from sklearn.preprocessing import StandardScaler

happy = pd.read_csv("data/world_happy_report.csv")

# will drop duplicates, next time can just do average of duplicates instead
happy = happy.drop_duplicates(subset=["Country", "Year"], keep="first")

# assign each country its own unique id
happy["Country_ID"] = happy["Country"].astype("category").cat.codes

columns_to_scale = [
    "GDP_per_Capita", "Social_Support", "Healthy_Life_Expectancy", 
    "Freedom", "Generosity", "Corruption_Perception", 
    "Unemployment_Rate", "Education_Index", "Population",
    "Urbanization_Rate", "Life_Satisfaction", "Public_Trust",
    "Mental_Health_Index", "Income_Inequality", "Public_Health_Expenditure",
    "Climate_Index", "Work_Life_Balance", "Internet_Access",
    "Crime_Rate", "Political_Stability", "Employment_Rate"
]

scaler = StandardScaler()
happy[columns_to_scale] = scaler.fit_transform(happy[columns_to_scale])

# rearranging columns
cols = ["Country_ID", "Country", "Year"] + \
       [col for col in happy.columns if col not in ["Country_ID", "Country", "Year"]]
happy = happy[cols]

happy.to_csv("data/preprocessed_happy.csv", index=False)
