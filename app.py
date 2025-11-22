from flask import Flask, render_template, request, url_for
import os
import matplotlib.pyplot as plt
import numpy as np
import joblib

app = Flask(__name__)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

RF_MODEL_PATH = os.path.join(BASE_DIR, "model/rf_model.pkl")


def load_rf_model():
    """
    Load model dengan aman.
    Jika model tidak kompatibel / tidak ditemukan,
    kembalikan None supaya app tetap jalan.
    """
    try:
        model = joblib.load(RF_MODEL_PATH)
        print("✅ Model RandomForest berhasil diload.")
        return model
    except FileNotFoundError:
        print("⚠️ Model tidak ditemukan. Pastikan file ada di model/rf_model.pkl")
        return None
    except Exception as e:
        # Biasanya error karena beda versi sklearn (seperti yang kamu alami)
        print("⚠️ Model gagal diload karena inkompatibel versi sklearn.")
        print(f"Detail error: {e}")
        return None


rf_model = load_rf_model()


def generate_plot(values, levels):
    plot_filename = "risk_plot.png"
    images_dir = os.path.join(BASE_DIR, "static/images")
    os.makedirs(images_dir, exist_ok=True)
    plot_path = os.path.join(images_dir, plot_filename)

    idx = np.arange(len(values))

    plt.figure(figsize=(8, 4))
    plt.plot(idx, levels, marker="o")
    plt.title("Grafik Prediksi Risiko Banjir")
    plt.xlabel("Fitur Input")
    plt.ylabel("Risiko (0-2)")
    plt.xticks(idx, ["Rainfall", "Water Level", "Soil Moisture", "Elevation"])
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()

    return plot_filename


def predict_levels(rainfall, waterlevel, soil, elevation):
    """
    Prediksi berbasis model jika ada.
    Kalau model None, pakai rule-based sederhana biar app tetap bisa dipakai.
    """
    if rf_model is None:
        # fallback sederhana (mirip logika training)
        if waterlevel <= 1.0:
            probs = np.array([1.0, 0.0, 0.0])
        elif waterlevel <= 2.5:
            probs = np.array([0.0, 1.0, 0.0])
        else:
            probs = np.array([0.0, 0.0, 1.0])
    else:
        inputs = np.array([[rainfall, waterlevel, soil, elevation]])
        probs = rf_model.predict_proba(inputs)[0]

        # jaga-jaga kalau model cuma 2 kelas
        if len(probs) == 2:
            probs = np.array([probs[0], probs[1], 0.0])

    levels = {
        "rendah": int(probs[0] * 2000),
        "sedang_rendah": int((probs[0] + probs[1]) / 2 * 2000),
        "sedang": int(probs[1] * 2000),
        "sedang_tinggi": int((probs[1] + probs[2]) / 2 * 2000),
        "tinggi": int(probs[2] * 2000),
    }
    return levels


@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    plot_url = None

    if request.method == "POST":
        rainfall = float(request.form["rainfall"])
        waterlevel = float(request.form["waterlevel"])
        soil = float(request.form["soil"])
        elevation = float(request.form["elevation"])

        prediction = predict_levels(rainfall, waterlevel, soil, elevation)

        values = [rainfall, waterlevel, soil, elevation]
        avg_level = np.mean(list(prediction.values()))
        levels = [avg_level] * 4

        filename = generate_plot(values, levels)
        plot_url = url_for("static", filename="images/" + filename)

    return render_template("index.html", prediction=prediction, plot_url=plot_url)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=True)
