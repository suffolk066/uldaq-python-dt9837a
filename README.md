# UL For Linux Python API Sample (WIP)
https://github.com/mccdaq/uldaq   
This project demonstrates how to use the UL for Linux Python API with the DT9837A Data Logger to collect and analyze data from sensors. Specifically, it focuses on acquiring analog input from a connected microphone and accelerometer.

![20240904-160352_output_all_in_one](https://github.com/user-attachments/assets/ac2cda4f-655c-4f50-a241-de81a4824d42)

## Analog Inputs
* Ain0: HS13M131 Accelerometers
* Ain1: EMM-82S-CTC Microphone

## Features (for demonstration purposes; results may not be accurate).
* Data Acquisition: Collects analog input data from the connected sensors.
* Data Filtering: Applies a high-pass filter to the acquired data to remove low-frequency noise.
* FFT Analysis: Performs a Fast Fourier Transform (FFT) to convert the time-domain data into the frequency domain.
* dB SPL Conversion: Converts the acquired data into sound pressure levels in dB SPL.

## How to Install
### 1. Install UL For Linux and Python API
```bash
sudo apt-get install libusb-1.0-0-dev

# Install UL For Linux
wget -N https://github.com/mccdaq/uldaq/releases/download/v1.2.1/libuldaq-1.2.1.tar.bz2
tar -xvjf libuldaq-1.2.1.tar.bz2

cd libuldaq-1.2.1
./configure && make
sudo make install

# Install UL For Linux Python
pip install uldaq
```
### 2. Install Python Dependencies
```bash
pip install -r requirements.txt
```
## Usage
To run the data acquisition and processing script, execute the following command:
```bash
python demonstrate.py
```

This script will perform the following operations:

1. Initialize the DAQ device and configure the input channels.
2. Start data acquisition from the specified channels.
3. Apply a high-pass filter to the collected data to remove low-frequency components.
4. Perform FFT on the filtered data to analyze its frequency components.
5. Convert the signal to dB SPL values for sound pressure level analysis.
6. Save the processed data to a CSV file for further analysis.

## Output
The script saves the collected and processed data to a CSV file in the output directory. The file is named using a timestamp to ensure uniqueness, and it contains the following information:

* Time: Timestamp of the measurement.
* Channel 0 Values: Raw sensor readings from the sensor connected to channel 0 (e.g., accelerometer).
* Channel 1 Values: Raw sensor readings from the sensor connected to channel 1 (e.g., microphone).

Additionally, the script generates charts based on the sensor data, including:
* Acceleration Data Chart: Shows the acceleration data to G and m/s^2.
![20240904-154656_output_accelerometer_data](https://github.com/user-attachments/assets/ac0e7ffe-c330-49ba-9239-38a350eabb77)
![20240904-154656_output_accelerometer_mps2_data](https://github.com/user-attachments/assets/72257ed2-03d4-47a1-9e5c-cd08141ca9fe)
* Microphone Data Chart: Displays the microphone readings over time.
![20240904-083725_output_data](https://github.com/user-attachments/assets/a0269b89-2115-4720-a6fe-ed9bfbb00c37)
* Filtered Data Chart: Shows the microphone data after applying the high-pass filter to remove low-frequency noise.
![20240904-083725_output_noise_db_spl](https://github.com/user-attachments/assets/ba7961c8-ed37-487d-b0db-5d9c294f6df6)
* FFT Data Chart: Shows the frequency components of the signal obtained through FFT analysis.
![20240904-083725_output_noise_fft](https://github.com/user-attachments/assets/db424aa4-83a3-4ab5-b843-c22a57beb276)
* dB SPL Chart: Shows the sound pressure levels in decibels calculated from the sensor data. (not accurate)
![20240904-083725_output_noise_fft_db_spl](https://github.com/user-attachments/assets/3aa10483-a413-4146-811a-2fc86af2c755)

These charts are automatically generated and saved as image files in the output directory alongside the CSV file.
