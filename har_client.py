import requests
import time
import sys
import json

# --- CONFIGURATION ---
PHONE_IP_ADDRESS = "10.86.231.160" # Your phone's IP
PHONE_PORT = 5000                   
# --- THIS URL MATCHES THE "without g" SENSOR ---
URL = f"http://{PHONE_IP_ADDRESS}:{PHONE_PORT}/get?accX&accY&accZ"

MODEL_SERVER_URL = "http://127.0.0.1:5000/predict_activity" 

WINDOW_SIZE = 80
data_buffer = [] 
is_connected = True 

print(f"Connecting to phyphox at {URL}...")
print(f"Will send data to model at {MODEL_SERVER_URL}")
print("On your phone: Run 'Acceleration (WITHOUT g)' and enable 'Allow remote access'.")
print("Press Ctrl+C to stop.")

try:
    while True:
        try:
            response = requests.get(url=URL, timeout=1)
            response.raise_for_status()
            data = response.json()

            if not is_connected:
                print("\n✅ Reconnected to phone!")
                is_connected = True
            
            # --- THESE KEYS MATCH THE "without g" SENSOR ---
            accX = data['buffer']['accX']['buffer'][0]
            accY = data['buffer']['accY']['buffer'][0]
            accZ = data['buffer']['accZ']['buffer'][0]
            
            if accX is None:
                print("Waiting for data... (Did you press PLAY in the app?)", end="\r", flush=True)
                time.sleep(0.5)
                continue

            data_buffer.append([accX, accY, accZ])

            if len(data_buffer) == WINDOW_SIZE:
                try:
                    post_data = {"readings": data_buffer}
                    requests.post(MODEL_SERVER_URL, json=post_data, timeout=1)
                except Exception as e:
                    print(f"\n!! Error sending to model server: {e}")
                    time.sleep(2)

                data_buffer = data_buffer[20:] 
            
            else:
                print(f"Buffering... {len(data_buffer)}/{WINDOW_SIZE}", end="\r", flush=True)

            time.sleep(0.05) 

        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
            if is_connected:
                print("\n❌ Connection to phone lost. Will keep trying...")
                is_connected = False
            
            print("Reconnecting...", end="\r", flush=True)
            time.sleep(2)
        
        except KeyError as e:
            # This error will be gone now
            print(f"\nError: Data key {e} not found. Are you running 'Acceleration (WITHOUT g)'?")
            data_buffer = [] 
            time.sleep(2)

except KeyboardInterrupt:
    print("\nStopping data collection.")