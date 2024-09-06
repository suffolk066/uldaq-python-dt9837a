import sys, os, csv, time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from uldaq import AInScanFlag, Range, AiInputMode, CouplingMode, get_daq_device_inventory, InterfaceType, DaqDevice, IepeMode, ScanOption, create_float_buffer, ScanStatus
from scipy.signal import butter, filtfilt

# Set channel: if you using only 1 sensor, connect sensor to ain0 and set both channel to 0 -> Example) LOW_CHANNEL = 0, HIGH_CHANNEL = 0
# if you using 2+ sensor, connect ain0, ain1, ... ainn and set channel LOW_CHANNEL = 0, HIGH_CHANNEL = n
LOW_CHANNEL = 0
HIGH_CHANNEL = 1

# Set sensitivity
ACCELEROMETER_SENSITIVITY = 0.09878
MICROPHONE_SENSITIVITY = 0.03

# sample rate in samples per channel per second.
SAMPLE_RATE = 1000

# Set max measurement time
MAX_MEASUREMENT_SECOND = 10

# Save Option
OUTPUT_PATH = 'output'
CSV_NAME_FORMAT = 'data_log.csv'

def get_timestamp() -> str:
    time = pd.Timestamp.strftime(pd.Timestamp.now(), '%Y%m%d-%H%M%S') # (str) 20240903-093742
    return time

class DataLogger():
    def __init__(self, timestamp) -> None:
        self.devices = get_daq_device_inventory(InterfaceType.ANY)
        self.daq_device = DaqDevice(self.devices[0])
        self.ai_device = self.daq_device.get_ai_device()
        self.ai_config = self.ai_device.get_config()
        self.status = ScanStatus.IDLE
        self.data = None
        self.actual_sample_rate = None
        self.timestamp = timestamp
        
        self.set_channel()
        self.init_setting()
        self.get_csv_writer()
    
    def set_channel(self):
        # Default Value: 1.0 volts per unit(V/unit)
        # Set sesitivity for V/Unit to mV/Unit
        self.configure_channel(LOW_CHANNEL, ACCELEROMETER_SENSITIVITY) # mV/g
        self.configure_channel(HIGH_CHANNEL, MICROPHONE_SENSITIVITY) # mV/Pa

    def configure_channel(self, channel: int, sensitivity_value: float):
        self.ai_config.set_chan_iepe_mode(channel, IepeMode.ENABLED)
        self.ai_config.set_chan_coupling_mode(channel, CouplingMode.AC)
        self.ai_config.set_chan_sensor_sensitivity(channel, sensitivity_value)

    def init_setting(self):
        # create buffer -> number_of_channels * samples_per_channel
        channel_count = HIGH_CHANNEL - LOW_CHANNEL + 1 # Channel Count 0 = 1 channel, 1 = 2 channel, ...
        samples_per_channel = 1000
        self.data = create_float_buffer(channel_count, samples_per_channel) # len(data) = 2 * 1000 = 2000

        self.daq_device.connect()
        self.actual_sample_rate = self.ai_device.a_in_scan(LOW_CHANNEL, HIGH_CHANNEL, AiInputMode.SINGLE_ENDED, Range.BIP10VOLTS, samples_per_channel, SAMPLE_RATE, ScanOption.CONTINUOUS, AInScanFlag.DEFAULT, self.data)

    def get_csv_writer(self):
        # Create csv file
        data_path = os.path.join(OUTPUT_PATH, self.timestamp)
        if not os.path.exists(OUTPUT_PATH):
            os.mkdir(OUTPUT_PATH)
        else:
            os.mkdir(data_path)

        csv_file_name = os.path.join(data_path, '_'.join([self.timestamp, CSV_NAME_FORMAT]))
        self.csv_file = open(csv_file_name, mode='w', newline='')
        self.csv_writer = csv.writer(self.csv_file)

        # Write Row
        self.csv_writer.writerow(['seconds', 'channel_0_value', 'channel_1_value'])

    def start_scan(self):
        try:
            start_time = time.time()
            next_sleep_time = time.perf_counter()
            elapsed_time = 0

            while True:
                self.status, transfer_status = self.ai_device.get_scan_status()
                index = transfer_status.current_index
                total_count = transfer_status.current_total_count
                scan_count = transfer_status.current_scan_count

                if index >= 0:
                    elapsed_time = time.time() - start_time
                    channel_0_value = self.data[index]
                    channel_1_value = self.data[index + 1]
                    # Write data
                    self.csv_writer.writerow([elapsed_time, channel_0_value, channel_1_value])
                    # Process log
                    sys.stdout.write(f'\nElapsed Time = {elapsed_time:.6f} | Actual scan rate = {self.actual_sample_rate:.6f} | Current index = {index} | Scan count = {scan_count} | Total count = {total_count}')

                if not MAX_MEASUREMENT_SECOND == 0:
                    if elapsed_time >= MAX_MEASUREMENT_SECOND:
                        sys.stdout.writelines(f'\nMeasurement stopped after {elapsed_time:.2f} seconds\n')
                        break
                else:
                    # if MAX_MEASUREMENT_SECOND == 0 -> Set continous -> WIP
                    pass

                next_sleep_time += 1.0 / SAMPLE_RATE
                time.sleep(max(0, next_sleep_time - time.perf_counter()))

        except (ValueError, NameError, SyntaxError) as e:
            print(e)
        except KeyboardInterrupt as e:
            print(e)
        finally:
            self.stop_scan()
            self.csv_file.close()

    def stop_scan(self):
        if self.daq_device:
            if self.status == ScanStatus.RUNNING:
                self.ai_device.scan_stop()
            if self.daq_device.is_connected():
                self.daq_device.disconnect()
            self.daq_device.release()

class DataVisualizationCSV:
    def __init__(self, timestamp: str, file_name: str = None) -> None:
        self.timestamp = timestamp
        self.file_path = file_name or f'{os.path.join(OUTPUT_PATH, timestamp)}/{timestamp}_{CSV_NAME_FORMAT}'
        self.data: pd.DataFrame = pd.read_csv(self.file_path)
        self.max_freq = SAMPLE_RATE
        self.reference_pressure = 20e-6  # 20 µPa in Pascals
        self.highpass_cutoff = 1.5 # 1.5 Hz
        self.nyquist_freq = 0.5 * SAMPLE_RATE
        self.rms_interval = 0.1

        # mV/Unit
        self.data['channel_0_g'] = self.data['channel_0_value'] * ACCELEROMETER_SENSITIVITY
        self.data['channel_1_pa'] = self.data['channel_1_value'] * MICROPHONE_SENSITIVITY

        # Get Acceleration Data
        self._calculate_acceleration()

        # Get FFT
        self.limited_frequencies, self.limited_magnitude = self._calculate_fft(self.data['channel_1_pa'])
        self.actual_max_freq = round(self.limited_frequencies[-1], 1)

        # Convert to dB SPL
        self.db_spl_data = self._calculate_db_spl_rms(self.data['channel_1_pa'])
        self.db_spl_magnitude = self._convert_to_db_spl(self.limited_magnitude)

    def _calculate_fft(self, data):
        sampling_rate = 1 / np.mean(np.diff(self.data['seconds']))
        n = len(data)
        fft_result = np.fft.fft(data)
        frequencies = np.fft.fftfreq(n, d=1/sampling_rate)

        positive_frequencies = frequencies[:n // 2]
        fft_magnitude = np.abs(fft_result)[:n // 2]

        if self.max_freq >= self.nyquist_freq:
            indices = positive_frequencies <= self.nyquist_freq
        else:
            indices = positive_frequencies <= self.max_freq
        return positive_frequencies[indices], fft_magnitude[indices]
    
    def _calculate_rms(self, data):
        return np.sqrt(np.mean(data ** 2))

    def _convert_to_db_spl(self, rms_value):
        return 20 * np.log10(rms_value / self.reference_pressure)

    def _calculate_db_spl_rms(self, value):
        # Group data by each <second> interval
        self.data['time_bin'] = (self.data['seconds'] // self.rms_interval).astype(int)
        db_spl_values = []
        db_spl_time = []

        grouped = self.data.groupby('time_bin')
        for _, group in grouped:
            window_data = group['channel_1_pa'].values

            # Calculate RMS
            rms_value = self._calculate_rms(window_data)

            # Convert RMS to dB SPL
            db_spl_value = self._convert_to_db_spl(rms_value)
            db_spl_values.append(db_spl_value)

            # Calculate time: use the mean time of each group
            mean_time = group['seconds'].mean()
            db_spl_time.append(mean_time)

        self.db_spl_data_time = np.array(db_spl_time)
        return np.array(db_spl_values)
    
    def _calculate_acceleration(self):
        self.data['channel_0_mps2'] = self.data['channel_0_g'] * 9.81

    def plot_all_in_one(self):
        # Create subplots (2 x 3 grid)
        fig, axs = plt.subplots(3, 2, figsize=(24, 12))
        fig.suptitle(f'All Data Plots for {self.timestamp}', fontsize=16)

        # Plot 1: Acceleration in G over Time
        axs[0, 0].plot(self.data['seconds'], self.data['channel_0_g'], color='blue', label='Acceleration (g)')
        axs[0, 0].set_title('Acceleration Over Time (g)')
        axs[0, 0].set_xlabel('Time (Seconds)')
        axs[0, 0].set_ylabel('Acceleration (g)')
        axs[0, 0].legend()
        axs[0, 0].grid(True)

        # Plot 2: Acceleration in m/s² over Time
        axs[0, 1].plot(self.data['seconds'], self.data['channel_0_mps2'], color='green', label='Acceleration (m/s²)')
        axs[0, 1].set_title('Acceleration Over Time (m/s²)')
        axs[0, 1].set_xlabel('Time (Seconds)')
        axs[0, 1].set_ylabel('Acceleration (m/s²)')
        axs[0, 1].legend()
        axs[0, 1].grid(True)

        # Plot 3: Microphone Data over Time
        axs[1, 0].plot(self.data['seconds'], self.data['channel_1_pa'], color='red', label='Sound Presure (Pa)')
        axs[1, 0].set_title('Sound Presure Over Time (Pa)')
        axs[1, 0].set_xlabel('Time (Seconds)')
        axs[1, 0].set_ylabel('Sound Presure (Pa)')
        axs[1, 0].legend()
        axs[1, 0].grid(True)

        # Plot 4: FFT of Microphone Data
        axs[1, 1].plot(self.limited_frequencies, self.limited_magnitude, color='purple')
        axs[1, 1].set_title(f'Frequency Spectrum (0-{self.actual_max_freq} Hz)')
        axs[1, 1].set_xlabel('Frequency (Hz)')
        axs[1, 1].set_ylabel('Amplitude')
        axs[1, 1].legend()
        axs[1, 1].grid(True)

        # Plot 5: Noise Sensor Data in dB SPL
        axs[2, 0].plot(self.db_spl_data_time, self.db_spl_data, color='orange', label='Sound Presure (dB SPL RMS)')
        axs[2, 0].set_title('Sound Presure Over Time(dB SPL RMS)')
        axs[2, 0].set_xlabel('Time (Seconds)')
        axs[2, 0].set_ylabel('Sound Presure (dB SPL RMS)')
        axs[2, 0].legend()
        axs[2, 0].grid(True)

        # Plot 6: FFT Data in dB SPL
        axs[2, 1].plot(self.limited_frequencies, self.db_spl_magnitude, color='purple', label='Amplitude')
        axs[2, 1].set_title(f'Amplitude in Frequency Domain (dB SPL, 0-{self.actual_max_freq} Hz)')
        axs[2, 1].set_xlabel('Frequency (Hz)')
        axs[2, 1].set_ylabel('Amplitude')
        axs[2, 1].legend()
        axs[2, 1].grid(True)

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        file_suffix = 'output_all_in_one'
        figure_name = f'{os.path.join(OUTPUT_PATH, self.timestamp)}/{self.timestamp}_{file_suffix}.png'
        plt.savefig(figure_name)
        
    #region single plot
    def _plot_single_data(self, x, y, title, xlabel, ylabel, file_suffix, colors='blue', labels=None):
        plt.figure(figsize=(12, 6))
        plt.plot(x, y, color=colors, label=labels)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        if labels:
            plt.legend()
        plt.grid(True)
        figure_name = f'{os.path.join(OUTPUT_PATH, self.timestamp)}/{self.timestamp}_{file_suffix}.png'
        plt.savefig(figure_name)

    #region accelerometer plot
    def plot_accelerometer(self):
        self._plot_single_data(
            self.data['seconds'], 
            self.data['channel_0_g'], 
            'Acceleration Data Over Time', 
            'Time (Secodns)', 
            'Acceleration (g)', 
            'output_accelerometer_data',
            labels='Acceleration (g)'
        )

    def plot_acceleration_mps2(self):
        self._plot_single_data(
            self.data['seconds'], 
            self.data['channel_0_mps2'], 
            'Acceleration Data Over Time (m/s²)', 
            'Time (Secodns)', 
            'Acceleration (m/s²)', 
            'output_accelerometer_mps2_data',
            labels='Acceleration (m/s²)'
        )
    #endregion

    #region microphone plot
    def plot_microphone_data(self):
        self._plot_single_data(
            self.data['seconds'], 
            self.data['channel_1_pa'], 
            'Sound Presure Over Time', 
            'Time (Seconds)', 
            'Sound Presure (Pa)', 
            'output_microphone_data', 
            labels='Sound Presure (Pa)'
        )

    def plot_fft(self):
        self._plot_single_data(
            self.limited_frequencies, 
            self.limited_magnitude, 
            f'Frequency Spectrum (0-{self.actual_max_freq} Hz)', 
            'Frequency (Hz)', 
            'Magnitude', 
            'output_microphone_fft', 
        )

    def plot_db_spl(self):
        self._plot_single_data(
            self.data['seconds'], 
            self.db_spl_data, 
            'Sound Presure Over Time (dB SPL RMS)', 
            'Time (Seconds)', 
            'dB SPL', 
            'output_microphone_db_spl', 
            labels='Sound Presure in dB SPL RMS'
        )

    def plot_fft_db_spl(self):
        self._plot_single_data(
            self.limited_frequencies, 
            self.db_spl_magnitude, 
            f'Amplitude in Frequency Domain (dB SPL, 0-{self.actual_max_freq} Hz)', 
            'Frequency (Hz)', 
            'Amplitude', 
            'output_microphone_fft_db_spl', 
            labels='Amplitude in Frequency Domain'
        )
    #endregion
    #endregion

    def run(self):
        self.plot_all_in_one()

        # If you want a single chart, use the function below.
        # self.plot_accelerometer()
        # self.plot_acceleration_mps2()
        # self.plot_microphone_data()
        # self.plot_fft()
        # self.plot_db_spl()
        # self.plot_fft_db_spl()

if __name__ == '__main__':
    timestamp = get_timestamp()
    logger = DataLogger(timestamp)
    logger.start_scan()
    noise_vis = DataVisualizationCSV(timestamp)
    noise_vis.run()