import streamlit as st
from PIL import Image
import os
import subprocess  # Added to run the external script

from input_image2 import extract_ecg_to_csv

def save_uploaded_file(uploaded_file):
    try:
        with open(os.path.join("uploaded_images", uploaded_file.name), "wb") as f:
            f.write(uploaded_file.getbuffer())
        return True
    except Exception as e:
        print(e)
        return False

# Function to run ed.py script and capture its output
def run_ed_script():
    result = subprocess.run(['python', 'ed.py'], capture_output=True, text=True)  # Running the ed.py script
    return result.stdout  # Returning the script output

def run_gan_script():
    result = subprocess.run(['python', 'gan.py'], capture_output=True, text=True)  # Running the ed.py script
    return result.stdout  # Returning the script output


# Create the Streamlit app
st.title("ECG Image Processor")

# File uploader
uploaded_file = st.file_uploader("Choose an ECG image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Save the uploaded file to disk
    if save_uploaded_file(uploaded_file):
        st.success("File uploaded successfully.")
        st.image(uploaded_file, caption="Uploaded ECG Image.", use_column_width=True)
        
        # File path for saving the processed CSV
        output_csv_path = "output_data.csv"
        
        # Call the function from input_image2.py
        extract_ecg_to_csv(uploaded_file.name, output_csv_path, num_points=141, amplitude_scale=1.0, visualize=False)
        
        st.success(f"ECG amplitude data successfully written to {output_csv_path}")
        st.download_button("Download CSV", data=open(output_csv_path).read(), file_name=output_csv_path, mime='text/csv')
        
        # Add button to run anomaly detection
        if st.button("Run Anomaly Detection"):  # Added button for anomaly detection
            result = run_ed_script()  # Run the ed.py script when button is clicked
            st.text(result)  # Display the output of ed.py

        # Add button to run anomaly detection
        if st.button("Run GAN"):  # Added button for anomaly detection
            result = run_gan_script()  # Run the ed.py script when button is clicked
            st.text(result)  # Display the output of ed.py
    else:
        st.error("Failed to upload file.")