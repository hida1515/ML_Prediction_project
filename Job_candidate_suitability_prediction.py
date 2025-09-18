import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier

# 1. Load dataset
df = pd.read_csv("job_candidates_large.csv")  # dataset must have 'Suitability' column (0/1)
print(df.head())

# 2. Features & Target
X = df.drop("Suitability", axis=1)
y = df["Suitability"]

# Identify categorical & numeric columns
categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

# Column transformer for preprocessing
ct = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ("num", StandardScaler(), numeric_cols),
    ]
)

# 3. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 4. Model training
X_train = ct.fit_transform(X_train)
X_test = ct.transform(X_test)

model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

print("\nModel trained successfully.")

# 5. User Input Prediction
print("\n--- Job Candidate Suitability Prediction ---")
age = int(input("Enter Age: "))
edu = input("Enter Education (Bachelors/Masters/PhD/etc): ")
exp = int(input("Enter Years of Experience: "))
skills = int(input("Enter Skill Score (1-10): "))
certs = int(input("Enter Certifications Count: "))
leadership = int(input("Enter Leadership Score (1-10): "))
comm = int(input("Enter Communication Score (1-10): "))
intern = int(input("Has Internship Experience? (1=Yes, 0=No): "))

# Prepare user input
user_df = pd.DataFrame([{
    "Age": age,
    "Education": edu,
    "Experience": exp,
    "Skills": skills,
    "Certifications": certs,
    "Leadership": leadership,
    "Communication": comm,
    "Internship": intern
}])

user_X = ct.transform(user_df)
prediction = model.predict(user_X)[0]

print(f"\nPredicted Result: {'Suitable for Job' if prediction == 1 else 'Not Suitable'}")
