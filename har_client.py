import requests
import time
import sys
import json

# --- CONFIGURATION ---
PHONE_IP_ADDRESS = "10.86.231.160" # Your phone's IP
PHONE_PORT = 5000                   
URL = f"http://{PHONE_IP_ADDRESS}:{PHONE_PORT}/get?accX&accY&accZ"

MODEL_SERVER_URL = "http://127.0.0.1:5000/predict_activity" 

WINDOW_SIZE = 80
data_buffer = [] 
is_connected = True # --- NEW: State variable to track connection

print(f"Connecting to phyphox at {URL}...")
print(f"Will send data to model at {MODEL_SERVER_URL}")
print("On your phone: Run 'Acceleration (without g)' and enable 'Allow remote access'.")
print("Press Ctrl+C to stop.")

try:
    while True:
        try:
            # 1. Fetch the data from the phone
            response = requests.get(url=URL, timeout=1)
            response.raise_for_status()
            data = response.json()

            # --- NEW: Print "Reconnected!" if we were previously disconnected ---
            if not is_connected:
                print("\n✅ Reconnected to phone!")
                is_connected = True
            
            # 2. Extract the latest sensor readings
            accX = data['buffer']['accX']['buffer'][0]
            accY = data['buffer']['accY']['buffer'][0]
            accZ = data['buffer']['accZ']['buffer'][0]
            
            if accX is None:
                print("Waiting for data... (Did you press PLAY in the app?)", end="\r", flush=True)
                time.sleep(0.5)
                continue

            # 3. Add data to our buffer
            data_buffer.append([accX, accY, accZ])

            # 4. Check if buffer is full
            if len(data_buffer) == WINDOW_SIZE:
                # print(f"Buffer full ({WINDOW_SIZE} samples). Sending to model...") # Too noisy
                
                try:
                    # 5. Send the full buffer to the model server
                    post_data = {"readings": data_buffer}
                    requests.post(MODEL_SERVER_URL, json=post_data, timeout=1)
                    
                except Exception as e:
                    # If the server is down, just print and keep trying
                    print(f"\n!! Error sending to model server: {e}")
                    time.sleep(2)

                # 6. SLIDE THE WINDOW
                data_buffer = data_buffer[20:] 
            
            else:
                # Just show we are collecting data
                print(f"Buffering... {len(data_buffer)}/{WINDOW_SIZE}", end="\r", flush=True)

            # Control fetch speed: 0.05 = 20Hz (matches your dataset)
            time.sleep(0.05) 

        # --- THIS IS THE MODIFIED PART ---
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
            if is_connected:
                print("\n❌ Connection to phone lost. Will keep trying...")
                is_connected = False
            
            print("Reconnecting...", end="\r", flush=True)
            # Just wait 2 seconds and let the 'while True' loop try again
            time.sleep(2)
        
        except KeyError as e:
            print(f"\nError: Phone sent bad data (KeyError: {e}). Clearing buffer and retrying.")
            data_buffer = [] # Clear buffer on bad data
            time.sleep(2)

except KeyboardInterrupt:
    print("\nStopping data collection.")