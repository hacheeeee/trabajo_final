# create_model.py
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import pickle
import os

os.makedirs("models", exist_ok=True)

# Generamos datos sintéticos parecidos a la guía:
# columnas: porcentaje_ciclistas, porcentaje_fumadores, porcentaje_enfermedad
rng = np.random.default_rng(42)
n = 500
porc_bici = rng.uniform(0, 50, size=n)
porc_fuma = rng.uniform(0, 50, size=n)
# objetivo sintético (solo para ejemplo)
y = 0.3 * porc_fuma + 0.1 * porc_bici + rng.normal(0, 3, size=n)

df = pd.DataFrame({
    "porcentaje_ciclistas": porc_bici,
    "porcentaje_fumadores": porc_fuma,
    "porcentaje_enfermedad": y
})

X = df[["porcentaje_ciclistas","porcentaje_fumadores"]].values
y = df["porcentaje_enfermedad"].values

model = RandomForestRegressor(n_estimators=50, random_state=42)
model.fit(X, y)

with open("models/model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Modelo creado y guardado en models/model.pkl")
