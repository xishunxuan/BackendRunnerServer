import requests
import json
import os
import time

# --- Configuration ---
# URL of the workstation server
WORKSTATION_URL = "http://localhost:7000/run-algorithm/"

# URL where the workstation server should send back file/status updates.
# You can use a service like https://webhook.site to get a test URL.
CALLBACK_URL = "https://webhook.site/6431a7f0-4694-4a8f-a424-78db1d18af6e" 

# Path to a dummy config file to send
CONF_FILE_PATH = "dummy_config.conf"

# Directory where the "algorithm" will create files.
# Make sure this path exists on the machine running the workstation_server.
# Use an absolute path.
OUTPUT_PATH = "D:/pythonprograms/BackendRunnerServer/test_output"
ENV = "pytorch112_py311"
# --- End of Configuration ---

def create_dummy_conf_file():
    """Creates a dummy .conf file for testing."""
    with open(CONF_FILE_PATH, "w") as f:
        f.write("[Settings]\n")
        f.write("param1 = value1\n")
        f.write("param2 = 123\n")
    print(f"Created dummy config file: {CONF_FILE_PATH}")

def run_test():
    """
    Sends a request to the workstation server to start the "algorithm".
    """
    if not os.path.exists(OUTPUT_PATH):
        os.makedirs(OUTPUT_PATH)
        print(f"Created output directory: {OUTPUT_PATH}")
        
    if CALLBACK_URL == "https://webhook.site/your-unique-id":
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print("!!! WARNING: Please update the CALLBACK_URL in this script !!!")
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        return

    create_dummy_conf_file()

    # Prepare the data for the POST request
    run_params = {
        "output_path": OUTPUT_PATH,
        "callback_url": CALLBACK_URL,
        "cuda_devices": "0", # Example GPU
        "env": ENV,
    }

    files = {
        'conf_file': (os.path.basename(CONF_FILE_PATH), open(CONF_FILE_PATH, 'rb'), 'text/plain')
    }
    
    data = {
        'run_params': json.dumps(run_params)
    }

    print("Sending request to workstation server...")
    
    try:
        response = requests.post(WORKSTATION_URL, files=files, data=data)
        response.raise_for_status() # Raise an exception for bad status codes
        
        print("Request successful!")
        print("Server response:", response.json())
        
        # Now, you can manually add/modify files in the `test_output` directory
        # to see if the workstation server sends them to your CALLBACK_URL.
        # Also, check the console output of the workstation server.

    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")



if __name__ == "__main__":
    run_test()
