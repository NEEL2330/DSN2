import subprocess
import threading
import uvicorn
import os

frontend_path = os.path.abspath("../potato-disease-frontend1")

def start_backend():
    uvicorn.run("main:app", host="127.0.0.1", port=8000)

def start_frontend():
    npm_path = "C:\\Program Files\\nodejs\\npm.cmd"  # Use .cmd here
    subprocess.run([npm_path, "start"], cwd=frontend_path)

if __name__ == "__main__":
    t1 = threading.Thread(target=start_backend)
    t2 = threading.Thread(target=start_frontend)

    t1.start()
    t2.start()

    t1.join()
    t2.join()
