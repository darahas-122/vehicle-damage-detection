# VehDetect | AI Vehicle Damage Detection

VehDetect is an AI-powered vehicle damage inspection web application trained on the CarDD dataset using an **EfficientNetB0** multi-label deep learning model. It detects 6 types of vehicle damage in a single photo:
- **Dent**
- **Scratch**
- **Crack**
- **Glass Shatter**
- **Lamp Broken**
- **Tire Flat**

---

## 🚀 Live Working URLs

- **Public Live URL (HTTPS)**: [https://ngtnw-2409-40f4-4478-fa35-8c94-ca83-f080-4764.free.pinggy.net](https://ngtnw-2409-40f4-4478-fa35-8c94-ca83-f080-4764.free.pinggy.net)
- **Local Access URL**: [http://localhost:5000](http://localhost:5000)

---

## 🛠️ Running Locally

Double-click `run_app.bat` or run:
```bash
.\.venv\python.exe run_with_tunnel.py
```
This starts the Flask AI backend, loads the Keras 3 model, serves the frontend UI, and establishes an active public HTTPS tunnel.
