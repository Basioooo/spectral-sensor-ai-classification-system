import os
import numpy as np
import pandas as pd
from flask import Flask, render_template_string, request
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report
from tensorflow import keras

app = Flask(__name__)

FILES = {
    'colors': "colors.csv",
    'materials': "materials.csv",
}

WAVELENGTH_COLUMNS = [
    "wavelength_410","wavelength_435","wavelength_460","wavelength_485",
    "wavelength_510","wavelength_535","wavelength_560","wavelength_585",
    "wavelength_610","wavelength_645","wavelength_680","wavelength_705",
    "wavelength_730","wavelength_760","wavelength_810","wavelength_860",
    "wavelength_900","wavelength_940"
]


model = None
scaler = None
label_encoder = None
train_acc = None
test_acc = None
class_report_text = None

# =========================
# Data Loading
# =========================
def load_all_data():
    frames = []
    for name, path in FILES.items():
        print(f"Loading {path}...")
        if not os.path.exists(path):
            print(f"❌ File not found: {path}")
            continue

        df = pd.read_csv(path)
        print("✅ Columns read:", df.columns.tolist()[:5], "...")  # show first few
        print("Loaded columns:", df.columns.tolist())

        df.columns = df.columns.str.strip().str.replace('\ufeff', '')
        print("✅ Columns after cleaning:", df.columns.tolist()[:5], "...")  # show first few

        # Check required columns
        missing = [col for col in WAVELENGTH_COLUMNS if col not in df.columns]
        if missing:
            print(f"❌ Missing columns in {path}: {missing}")
            continue

        if "label" not in df.columns:
            print(f"❌ No label column found in {path}")
            continue

        df = df[WAVELENGTH_COLUMNS + ["label"]]
        frames.append(df)


    if not frames:
        raise ValueError("No valid data loaded. Check your CSV headers.")

    return pd.concat(frames, axis=0, ignore_index=True)


# =========================
# Model
# =========================
def build_model(input_dim, num_classes):
    model = keras.Sequential([
        keras.layers.Input(shape=(input_dim, 1)),
        keras.layers.Conv1D(64, kernel_size=3, activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPooling1D(pool_size=2),
        keras.layers.Dropout(0.3),

        keras.layers.Conv1D(128, kernel_size=3, activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPooling1D(pool_size=2),
        keras.layers.Dropout(0.3),

        keras.layers.Flatten(),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# =========================
# Routes
# =========================
@app.route('/')
def home():
    status = "Model not trained yet." if model is None else "Model is trained and ready."
    return render_template_string("""
    <html><head><title>Spectral Classifier</title></head>
    <body style="background:linear-gradient(135deg,#4b0082,#8a2be2);color:#f0e6ff;font-family:Segoe UI;padding:24px;">
      <h2>🌈 Spectral Classifier Dashboard</h2>
      <div style="background:#1e1e1e;padding:20px;border-radius:12px;">
        <p><b>Status:</b> {{status}}</p>
        <a href="/train" style="background:#bb86fc;color:#121212;padding:10px 20px;border-radius:8px;text-decoration:none;">⚡ Train Model</a>
        <a href="/analyze" style="background:#bb86fc;color:#121212;padding:10px 20px;border-radius:8px;text-decoration:none;">📊 Analyze Dataset</a>
      </div>
      {% if train_acc %}
      <div style="margin-top:20px;">
        <h3>Metrics</h3>
        <p><b>Train Accuracy:</b> {{train_acc}}</p>
        <p><b>Test Accuracy:</b> {{test_acc}}</p>
        <pre>{{class_report_text}}</pre>
      </div>
      {% endif %}
      <div style="margin-top:20px;">
        <h3>Try a Prediction</h3>
        <form method="post" action="/predict">
          <p>Enter 18 comma-separated values (410 → 940nm):</p>
          <textarea name="reading" rows="4" style="width:100%;"></textarea><br/><br/>
          <button type="submit" style="background:#bb86fc;color:#121212;padding:10px 20px;border-radius:8px;">🔮 Predict</button>
        </form>
      </div>
    </body></html>
    """, status=status, train_acc=train_acc, test_acc=test_acc, class_report_text=class_report_text)

@app.route('/train')
def train():
    global model, scaler, label_encoder, train_acc, test_acc, class_report_text
    data = load_all_data()
    X = data[[f"wavelength_{c}" for c in WAVELENGTH_COLUMNS]].values.astype(np.float32)
    y = data['label'].values

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    num_classes = len(label_encoder.classes_)
    y_onehot = keras.utils.to_categorical(y_encoded, num_classes=num_classes)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X).reshape(-1, len(WAVELENGTH_COLUMNS), 1)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_onehot, test_size=0.2, random_state=42, stratify=y_encoded
    )

    model = build_model(input_dim=len(WAVELENGTH_COLUMNS), num_classes=num_classes)
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=40, batch_size=32, verbose=0)

    train_acc = round(model.evaluate(X_train, y_train, verbose=0)[1], 4)
    test_acc = round(model.evaluate(X_test, y_test, verbose=0)[1], 4)

    y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
    y_true = np.argmax(y_test, axis=1)
    class_report_text = classification_report(y_true, y_pred, target_names=label_encoder.classes_, digits=4)

    # Save trained model here
    model.save("spectral_model.h5")

    return home()

@app.route('/predict', methods=['POST'])
def predict():
    global model, scaler, label_encoder
    if model is None:
        return "Model not trained yet. Visit /train first."

    raw = request.form.get('reading', '').strip()
    try:
        values = [float(x) for x in raw.split(',')]
        if len(values) != len(WAVELENGTH_COLUMNS):
            return f"Expected {len(WAVELENGTH_COLUMNS)} values, got {len(values)}."
    except:
        return "Invalid input. Please enter 18 comma-separated numbers."

    X = np.array([values], dtype=np.float32)
    X_scaled = scaler.transform(X).reshape(-1, len(WAVELENGTH_COLUMNS), 1)
    pred = model.predict(X_scaled, verbose=0)[0]
    label = label_encoder.inverse_transform([np.argmax(pred)])[0]
    confidence = round(np.max(pred) * 100, 2)

    return render_template_string(f"""
    <html><body style="background:#121212;color:#f0e6ff;padding:24px;font-family:Segoe UI;">
      <div style="background:#1e1e1e;padding:20px;border-radius:12px;">
        <h3>Prediction Result</h3>
        <p><b>Predicted Class:</b> {label}</p>
        <p><b>Confidence:</b> {confidence}%</p>
        <a href="/" style="background:#bb86fc;color:#121212;padding:10px 20px;border-radius:8px;text-decoration:none;">⬅ Back</a>
      </div>
    </body></html>
    """)

@app.route('/analyze')
def analyze():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns
    import io, base64

    data = load_all_data()
    plots, summaries = [], []

    def plot_and_encode(fig):
        buf = io.BytesIO()
        plt.tight_layout()
        plt.savefig(buf, format="png")
        buf.seek(0)
        plots.append(base64.b64encode(buf.getvalue()).decode('utf-8'))
        buf.close()
        plt.close(fig)

    # Class distribution
    fig, ax = plt.subplots()
    counts = data['label'].value_counts()
    counts.plot(kind='bar', ax=ax, color='skyblue')
    ax.set_title("Class Distribution")
    summaries.append(counts.to_string())
    plot_and_encode(fig)

    # Average spectrum per class
    fig, ax = plt.subplots()
    means = data.groupby('label')[WAVELENGTH_COLUMNS].mean()
    for label, row in means.iterrows():
        ax.plot(WAVELENGTH_COLUMNS, row.values, label=label)
    ax.set_title("Average Spectrum per Class")
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Mean Intensity")
    ax.legend()
    summaries.append(means.round(2).to_string())
    plot_and_encode(fig)

    # Variance per wavelength
    fig, ax = plt.subplots()
    variances = data[WAVELENGTH_COLUMNS].var()
    variances.plot(kind='bar', ax=ax, color='orange')
    ax.set_title("Variance per Wavelength")
    ax.set_ylabel("Variance")
    summaries.append(variances.round(4).to_string())
    plot_and_encode(fig)

    # Correlation heatmap
    fig, ax = plt.subplots(figsize=(10,8))
    corr = data[WAVELENGTH_COLUMNS].corr()
    sns.heatmap(corr, cmap="coolwarm", annot=False, ax=ax)
    ax.set_title("Correlation Between Wavelengths")
    summaries.append(corr.round(2).to_string())
    plot_and_encode(fig)

    return render_template_string("""
    <html>
    <head>
      <title>Dataset Analysis</title>
      <style>
        body { background: linear-gradient(135deg, #4b0082, #8a2be2); color: #f0e6ff; font-family: 'Segoe UI'; padding: 20px; }
        h2 { color: #e0b3ff; }
        .card { background-color: rgba(30,30,30,0.85); border-radius: 12px; padding: 20px; margin-bottom: 20px; box-shadow: 0 4px 12px rgba(0,0,0,0.6); }
        img { max-width: 100%; border-radius: 8px; margin-bottom: 10px; }
        pre { background-color: #1e1e1e; padding: 10px; border-radius: 8px; overflow-x: auto; color: #f0e6ff; }
        a { background-color: #bb86fc; color: #121212; padding: 10px 20px; border-radius: 8px; text-decoration: none; font-weight: bold; transition: transform 0.2s ease; }
        a:hover { background-color: #9a67ea; transform: scale(1.05); }
      </style>
    </head>
    <body>
      <h2>Dataset Analysis Dashboard</h2>
      <div class="card"><a href="/">⬅ Back to Home</a></div>
      {% for plot, summary in zip(plots, summaries) %}
        <div class="card">
          <img src="data:image/png;base64,{{plot}}" />
          <pre>{{summary}}</pre>
        </div>
      {% endfor %}
    </body>
    </html>
    """, plots=plots, summaries=summaries, zip=zip)
    
if __name__ == '__main__':
    app.run(debug=True)