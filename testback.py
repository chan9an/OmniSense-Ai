import requests
import time
import sys
import json

# --- CONFIGURATION ---
# !!! EDIT THIS - This must be your phone's IP address
PHONE_IP_ADDRESS = "10.86.231.160" 

# Phyphox port (default is 8080, but you had 5000)
PHONE_PORT = 5000                   
URL = f"http://{PHONE_IP_ADDRESS}:{PHONE_PORT}/get?accX&accY&accZ"

# This is your new server's address (it's on the same PC)
MODEL_SERVER_URL = "http://127.0.0.1:5000/predict"

WINDOW_SIZE = 80 # Must match your model
data_buffer = [] # This will store our 80 samples

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
            
            # 2. Extract the latest sensor readings
            accX = data['buffer']['accX']['buffer'][0]
            accY = data['buffer']['accY']['buffer'][0]
            accZ = data['buffer']['accZ']['buffer'][0]
            
            if accX is None:
                print("Waiting for data... (Did you press PLAY in the app?)")
                time.sleep(0.5)
                continue

            # 3. Add data to our buffer
            data_buffer.append([accX, accY, accZ])

            # 4. Check if buffer is full
            if len(data_buffer) == WINDOW_SIZE:
                print(f"Buffer full ({WINDOW_SIZE} samples). Sending to model...")
                
                try:
                    # 5. Send the full buffer to the model server
                    post_data = {"readings": data_buffer}
                    model_response = requests.post(MODEL_SERVER_URL, json=post_data, timeout=1)
                    
                    # Print the prediction from the server
                    print(f"--> MODEL PREDICTS: {model_response.json()}")

                except Exception as e:
                    print(f"!! Error sending to model: {e}")

                # 6. SLIDE THE WINDOW: Remove the oldest 20 samples
                # This makes it predict every 20 new samples.
                data_buffer = data_buffer[20:] 
            
            else:
                # Just show we are collecting data
                print(f"Buffering... {len(data_buffer)}/{WINDOW_SIZE}", end="\\r")

            # Control fetch speed: 0.05 = 20Hz (matches your dataset)
            time.sleep(0.05) 

        except requests.exceptions.ConnectionError:
            print(f"\\nError: Could not connect to {URL}.")
            print("Check phone IP, Wi-Fi, and 'Allow remote access'.")
            sys.exit(1)
            
        except requests.exceptions.Timeout:
            print("Request timed out. Retrying...")
        
        except KeyError as e:
            print(f"\\nError: Data key {e} not found in {data.get('buffer', {}).keys()}")
            print("Make sure you are running 'Acceleration (without g)' in Phyphox.")
            time.sleep(2)

except KeyboardInterrupt:
    print("\\nStopping data collection.")