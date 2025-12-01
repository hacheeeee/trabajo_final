# app.py
from flask import Flask, render_template, request
import pickle
import numpy as np
import os

app = Flask(__name__)

# Ruta al modelo
MODEL_PATH = os.path.join("models", "model.pkl")

# Cargar el modelo una sola vez al iniciar la app
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Modelo no encontrado en {MODEL_PATH}. ")
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

@app.route("/", methods=["GET"])
def index():
    # Mostrar la plantilla sin resultado al inicio
    return render_template("index.html", prediction=None)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Obtener valores del formulario
        bici = request.form.get("porcentaje_ciclistas", "")
        fuma = request.form.get("porcentaje_fumadores", "")

        # Validaciones básicas
        if bici == "" or fuma == "":
            return render_template("index.html", prediction="Por favor ingresa ambos porcentajes.")

        # Convertir a float
        bici_val = float(bici)
        fuma_val = float(fuma)

        # Preparar entrada para el modelo: forma (1,2)
        X = np.array([[bici_val, fuma_val]])

        # Predicción
        pred = model.predict(X)[0]

        # Formatear resultado
        resultado = round(float(pred), 3)

        return render_template("index.html", prediction=f"Predicción (porcentaje estimado de enfermedad cardíaca): {resultado}")

    except ValueError:
        return render_template("index.html", prediction="Valores numéricos inválidos.")
    except Exception as e:
        # Mostrar error simple en la UI para depuración
        return render_template("index.html", prediction=f"Error en el servidor: {e}")

if __name__ == "__main__":
    # Modo debug para desarrollo
    app.run(debug=True)
