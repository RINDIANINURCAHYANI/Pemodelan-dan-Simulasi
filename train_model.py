import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

csv_file = "uploads/flood_prediction.csv"
if not os.path.exists(csv_file):
    raise FileNotFoundError(f"{csv_file} tidak ditemukan.")

output_folder = "model"
os.makedirs(output_folder, exist_ok=True)
model_path = os.path.join(output_folder, "rf_model.pkl")

features = ["Rainfall_mm","WaterLevel_m","SoilMoisture_pct","Elevation_m"]

def generate_risk(wl):
    if wl <= 1.0: return 0
    elif wl <= 2.5: return 1
    else: return 2

df = pd.read_csv(csv_file)
df['FloodRisk'] = df['WaterLevel_m'].apply(generate_risk)

X = df[features]
y = df['FloodRisk']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=300, max_depth=10, random_state=42)
model.fit(X_train, y_train)

joblib.dump(model, model_path)
print(f"Model berhasil disimpan di {model_path}")
print(f"Akurasi model: {model.score(X_test, y_test)*100:.2f}%")
