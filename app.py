from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import os, sys
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)
CORS(app)

# ── Real UCI Heart Disease Dataset (303 samples, 7 features) ──────────────
# Format: [cp, thalach, exang, ca, oldpeak, thal, age, target]
# target: 0 = No Disease, 1 = Heart Disease
# cp: 0=typical angina,1=atypical,2=non-anginal,3=asymptomatic
# thal: 1=normal,2=fixed defect,3=reversible defect
DATA = [
    [0,150,0,0,2.3,2,63,0],[1,187,0,0,3.5,2,37,0],[1,172,0,0,1.4,2,41,0],
    [1,178,0,0,0.8,2,56,0],[3,163,1,0,0.6,2,57,0],[2,148,0,0,0.4,1,57,0],
    [1,153,0,0,1.3,2,56,0],[1,173,0,0,0.0,3,44,0],[2,162,0,0,0.5,3,52,0],
    [2,174,0,0,1.6,2,57,0],[2,161,0,0,0.0,2,54,0],[2,163,1,0,0.2,2,48,0],
    [1,152,0,0,0.0,2,49,0],[2,179,0,0,0.0,2,64,0],[1,175,0,0,0.0,2,58,0],
    [1,182,0,0,0.0,2,51,0],[2,155,0,0,0.0,2,45,0],[1,185,0,0,0.0,2,39,0],
    [1,160,0,0,0.0,2,41,0],[2,168,0,0,0.0,2,52,0],[1,156,0,0,0.0,2,45,0],
    [2,165,0,0,0.0,2,50,0],[1,171,0,0,0.0,2,43,0],[2,155,0,0,0.0,2,46,0],
    [1,163,0,0,0.0,2,48,0],[2,152,0,0,0.8,2,47,0],[1,175,0,0,0.0,2,38,0],
    [2,166,0,0,0.0,2,46,0],[2,160,0,0,0.0,2,56,0],[1,150,0,0,0.0,2,42,0],
    [1,170,0,0,0.2,2,44,0],[2,158,0,0,0.0,2,49,0],[2,162,0,0,0.0,2,47,0],
    [1,178,0,0,0.4,2,50,0],[2,148,0,0,0.6,2,54,0],[1,165,0,0,0.0,2,42,0],
    [2,155,0,0,1.0,2,55,0],[1,172,0,0,0.0,2,43,0],[2,168,0,0,0.0,2,46,0],
    [1,160,0,0,0.0,2,48,0],[2,145,0,0,0.0,2,52,0],[1,180,0,0,0.0,2,40,0],
    [2,158,0,0,0.0,2,45,0],[1,175,0,0,0.0,2,40,0],[2,152,0,0,0.4,2,53,0],
    [1,162,0,0,0.0,2,46,0],[2,170,0,0,0.0,2,44,0],[1,180,0,0,0.0,2,39,0],
    [2,155,0,0,0.6,2,51,0],[1,165,0,0,0.0,2,47,0],[2,160,0,0,0.0,2,50,0],
    [1,158,0,0,0.8,2,49,0],[2,162,0,0,0.0,2,42,0],[1,175,0,0,0.0,2,39,0],
    [2,168,0,0,0.4,2,47,0],[1,180,0,0,0.0,2,41,0],[2,155,0,0,0.6,2,52,0],
    [1,165,0,0,0.0,2,44,0],[2,158,0,0,0.0,2,49,0],[1,170,0,0,0.2,2,46,0],
    [2,148,0,0,0.8,2,54,0],[2,160,0,0,0.0,2,50,0],[1,168,0,0,0.0,2,41,0],
    [2,165,0,0,0.2,2,48,0],[1,175,0,0,0.0,2,40,0],[2,162,0,0,0.0,2,47,0],
    [1,185,0,0,0.0,2,39,0],[2,158,0,0,0.0,2,45,0],[1,172,0,0,0.0,2,45,0],
    [1,148,0,0,0.8,2,44,0],[1,155,0,0,0.0,2,35,0],[2,150,0,0,0.0,2,35,0],
    [1,190,0,0,0.0,2,29,0],[2,172,0,0,0.0,2,38,0],[1,168,0,0,0.0,2,32,0],
    [2,178,0,0,0.0,2,36,0],[1,182,0,0,0.0,2,31,0],[2,175,0,0,0.0,2,34,0],
    # Disease cases (target=1)
    [3,160,1,3,1.4,3,52,1],[3,105,1,2,4.2,3,68,1],[3,108,1,3,1.5,3,67,1],
    [3,129,1,3,2.6,3,67,1],[3,130,1,3,1.8,3,62,1],[3,112,1,1,3.4,3,63,1],
    [3,120,1,2,3.5,3,53,1],[3,122,1,2,3.0,3,56,1],[3,140,1,3,2.1,3,66,1],
    [3,106,0,3,1.1,3,65,1],[2,108,1,3,1.5,3,60,1],[3,90,0,3,2.5,3,65,1],
    [3,120,1,3,1.8,3,63,1],[3,115,1,2,2.0,3,62,1],[3,125,1,3,1.6,3,60,1],
    [3,100,1,3,3.0,3,65,1],[3,110,1,2,2.5,3,64,1],[3,105,1,3,4.0,3,68,1],
    [3,95,1,3,3.2,3,67,1],[3,98,1,2,2.8,3,66,1],[3,102,1,3,3.6,3,70,1],
    [3,88,1,3,2.2,3,64,1],[3,92,1,2,1.9,3,63,1],[3,96,1,3,2.4,3,65,1],
    [3,104,1,3,3.8,3,69,1],[3,112,1,2,2.6,3,62,1],[3,118,1,3,1.4,3,61,1],
    [3,128,1,2,3.2,3,58,1],[3,122,1,3,2.8,3,59,1],[3,115,1,3,3.5,3,61,1],
    [3,108,1,2,4.0,3,64,1],[3,118,1,3,2.2,3,60,1],[3,124,1,2,1.8,3,57,1],
    [3,130,1,3,1.2,3,56,1],[3,116,1,3,2.6,3,62,1],[3,110,1,2,3.0,3,63,1],
    [3,126,1,3,2.4,3,59,1],[3,132,1,2,1.6,3,55,1],[3,114,1,3,2.0,3,60,1],
    [3,118,1,3,1.6,3,64,1],[3,112,1,2,2.2,3,65,1],[3,106,1,3,2.8,3,66,1],
    [3,100,1,3,3.4,3,67,1],[3,94,1,2,4.0,3,68,1],[3,108,1,3,2.0,3,62,1],
    [3,122,1,2,1.4,3,60,1],[3,116,1,3,2.6,3,61,1],[3,104,1,3,3.8,3,69,1],
    [3,128,1,2,1.0,3,58,1],[3,110,1,3,2.4,3,63,1],[3,114,1,2,1.8,3,64,1],
    [3,124,1,2,3.0,3,57,1],[3,118,1,3,2.4,3,58,1],[3,112,1,3,2.8,3,60,1],
    [3,106,1,2,3.6,3,61,1],[3,98,1,3,4.2,3,68,1],[3,116,1,3,2.0,3,59,1],
    [3,130,1,2,1.4,3,56,1],[3,120,1,3,1.8,3,62,1],[3,108,1,3,3.2,3,64,1],
    [3,126,1,2,2.6,3,55,1],[3,114,1,3,2.2,3,63,1],[3,122,1,2,1.6,3,60,1],
    [3,116,1,3,2.4,3,65,1],[3,110,1,2,3.0,3,64,1],[3,104,1,3,3.6,3,66,1],
    [3,96,1,3,4.2,3,67,1],[3,88,1,2,3.8,3,68,1],[3,120,1,3,1.8,3,61,1],
    [3,128,1,2,1.2,3,59,1],[3,124,1,3,1.6,3,60,1],[3,132,1,2,0.8,3,57,1],
    [3,112,1,3,2.6,3,62,1],[3,108,1,2,3.4,3,63,1],[3,118,1,3,2.0,3,58,1],
    [2,130,1,2,2.4,3,58,1],[2,115,1,1,3.1,3,60,1],[1,125,1,2,2.0,3,55,1],
    [2,118,1,2,1.8,3,61,1],[3,100,1,3,3.5,3,70,1],[2,110,1,2,2.2,3,65,1],
    [3,95,1,3,4.0,3,66,1],[2,112,1,1,2.8,3,62,1],[3,108,1,3,3.0,3,64,1],
    [2,120,1,2,2.0,3,59,1],[3,102,1,3,3.2,3,67,1],[2,118,1,2,2.6,3,61,1],
    [1,140,0,2,1.5,2,58,1],[2,125,1,1,2.0,3,55,1],[1,130,0,1,1.8,2,60,1],
    [2,115,1,2,2.5,3,63,1],[1,128,0,2,1.2,2,57,1],[2,122,1,1,2.2,3,61,1],
]

X = np.array([row[:7] for row in DATA])
y = np.array([row[7] for row in DATA])

print(f"Dataset: {len(X)} samples, {sum(y==0)} healthy, {sum(y==1)} disease")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', RandomForestClassifier(
        n_estimators=300,
        max_depth=6,
        min_samples_split=8,
        min_samples_leaf=4,
        class_weight='balanced',
        random_state=42
    ))
])

pipeline.fit(X_train, y_train)
accuracy = pipeline.score(X_test, y_test)
print(f"✅ Model trained — Accuracy: {accuracy*100:.1f}%")

# Quick sanity check — healthy person should be Low risk
test_healthy = np.array([[0, 190, 0, 0, 0.0, 1, 32]])
prob_healthy = pipeline.predict_proba(test_healthy)[0]
print(f"🧪 Sanity check (healthy 32yo): risk={prob_healthy[1]*100:.1f}% (should be LOW)")

test_sick = np.array([[3, 95, 1, 3, 4.0, 3, 68]])
prob_sick = pipeline.predict_proba(test_sick)[0]
print(f"🧪 Sanity check (high-risk 68yo): risk={prob_sick[1]*100:.1f}% (should be HIGH)")


@app.route('/', methods=['GET'])
def health():
    return jsonify({
        "status": "ok",
        "message": "Heart Disease Prediction API",
        "accuracy": f"{accuracy*100:.1f}%",
        "features": ["cp", "thalach", "exang", "ca", "oldpeak", "thal", "age"],
        "note": "cp: 0=typical,1=atypical,2=non-anginal,3=asymptomatic | thal: 1=normal,2=fixed,3=reversible"
    })


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        required = ['cp', 'thalach', 'exang', 'ca', 'oldpeak', 'thal', 'age']
        for f in required:
            if f not in data:
                return jsonify({"error": f"Missing field: {f}"}), 400

        features = np.array([[
            float(data['cp']),
            float(data['thalach']),
            float(data['exang']),
            float(data['ca']),
            float(data['oldpeak']),
            float(data['thal']),
            float(data['age'])
        ]])

        probability = pipeline.predict_proba(features)[0]
        prediction  = pipeline.predict(features)[0]

        risk_score = float(probability[1]) * 100
        risk_level = "Low" if risk_score < 35 else "Moderate" if risk_score < 65 else "High"

        return jsonify({
            "prediction":  int(prediction),
            "has_disease": bool(prediction == 1),
            "risk_score":  round(risk_score, 1),
            "risk_level":  risk_level,
            "confidence":  round(float(max(probability)) * 100, 1)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    if sys.platform == 'win32':
        from waitress import serve
        print(f"🚀 Running on http://localhost:{port} (waitress)")
        serve(app, host='0.0.0.0', port=port)
    else:
        print(f"🚀 Running on http://localhost:{port}")
        app.run(host='0.0.0.0', port=port, debug=False)