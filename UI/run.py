import subprocess
import os
import signal
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = os.path.join(BASE_DIR, "backend")
FRONTEND_DIR = os.path.join(BASE_DIR, "react-app", "sentiment-analysis-app")

NODEJS_PATH = r"C:\Program Files\nodejs"

os.environ["PATH"] += f";{NODEJS_PATH}"

backend = subprocess.Popen(
    ["python", "-m", "uvicorn", "api:app", "--reload"],
    cwd=BACKEND_DIR,
    shell=True
)

frontend = subprocess.Popen(
    ["npm", "run", "dev"],
    cwd=FRONTEND_DIR,
    shell=True
)

print("Backend (FastAPI) and Frontend (Vite) are running. Press Ctrl+C to stop.")

try:
    backend.wait()
    frontend.wait()
except KeyboardInterrupt:
    print("\nStopping servers...")
    backend.send_signal(signal.SIGINT)
    frontend.send_signal(signal.SIGINT)
    sys.exit(0)