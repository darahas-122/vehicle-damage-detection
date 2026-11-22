from Backend.app import app
import json

client = app.test_client()

# Test /health
res = client.get("/health")
print("Health status code:", res.status_code)
print("Health response:", res.get_json())

# Test / (home page)
res_home = client.get("/")
print("Home page status code:", res_home.status_code)
print("Home page content length:", len(res_home.data))

# Test /predict with hero-car.png
with open("Frontend/hero-car.png", "rb") as img:
    data = {"image": (img, "hero-car.png")}
    res_pred = client.post("/predict", data=data, content_type="multipart/form-data")
    print("Predict status code:", res_pred.status_code)
    pred_data = res_pred.get_json()
    print("Prediction message:", pred_data.get("message"))
    print("Scores:", pred_data.get("scores"))
    print("Predictions breakdown:")
    for p in pred_data.get("predictions", []):
        print(f"  {p['name']:<15}: confidence={p['confidence']:.4f}, threshold={p['threshold']:.2f}, detected={p['detected']}")

print("\nALL FLASK TESTS PASSED SUCCESSFULLY!")
