import subprocess
import threading
import time
import sys
import os
import re

# Set environment
os.environ.setdefault("KERAS_BACKEND", "torch")
keras_home = os.path.abspath(".keras")
os.environ.setdefault("KERAS_HOME", keras_home)
os.environ.setdefault("MPLCONFIGDIR", os.path.join(keras_home, "mpl"))

python_exe = os.path.abspath(r".venv\python.exe")
cloudflared_exe = os.path.abspath("cloudflared.exe")

print("=" * 60)
print("Starting VehDetect AI Server on port 5000...")
print("=" * 60)

# Start Flask app
app_process = subprocess.Popen(
    [python_exe, "Backend/app.py"],
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
    bufsize=1
)

def monitor_flask():
    for line in iter(app_process.stdout.readline, ''):
        print(f"[Flask] {line.strip()}", flush=True)

threading.Thread(target=monitor_flask, daemon=True).start()

# Wait for server to load model and start listening
time.sleep(5)

print("\n" + "=" * 60)
print("Starting Cloudflare Public Tunnel (https://*.trycloudflare.com)...")
print("=" * 60)

tunnel_cmd = [
    cloudflared_exe,
    "tunnel",
    "--url", "http://127.0.0.1:5000"
]

tunnel_process = subprocess.Popen(
    tunnel_cmd,
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
    bufsize=1
)

found_url = False
def monitor_tunnel():
    global found_url
    for line in iter(tunnel_process.stdout.readline, ''):
        line_clean = line.strip()
        match = re.search(r'https://[a-zA-Z0-9\-]+\.trycloudflare\.com', line)
        if match and not found_url:
            found_url = True
            url = match.group(0)
            print("\n" + "#" * 65, flush=True)
            print(f"🎉 CLOUDFLARE LIVE PUBLIC URL: {url}", flush=True)
            print(f"🏠 LOCAL ACCESS URL:           http://localhost:5000", flush=True)
            print("#" * 65 + "\n", flush=True)
            with open("deployment_url.txt", "w", encoding="utf-8") as uf:
                uf.write(f"PUBLIC_URL={url}\nLOCAL_URL=http://localhost:5000\n")
        elif "trycloudflare.com" in line_clean or "Registered tunnel connection" in line_clean:
            print(f"[Cloudflare] {line_clean}", flush=True)

threading.Thread(target=monitor_tunnel, daemon=True).start()

# Keep running
try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print("Shutting down...")
    app_process.terminate()
    tunnel_process.terminate()
