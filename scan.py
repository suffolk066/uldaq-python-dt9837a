import sys, os, csv, time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from uldaq import AInScanFlag, Range, AiInputMode, CouplingMode, get_daq_device_inventory, InterfaceType, DaqDevice, IepeMode, ScanOption, create_float_buffer, ScanStatus

# Set channel: if you using only 1 sensor, connect sensor to ain0 and set both channel to 0 -> Example) LOW_CHANNEL = 0, HIGH_CHANNEL = 0
# if you using 2+ sensor, connect ain0, ain1, ... ainn and set channel LOW_CHANNEL = 0, HIGH_CHANNEL = n
LOW_CHANNEL = 0
HIGH_CHANNEL = 1

# Set sensitivity
VIBRATION_SENSOR_SENSITIVITY = 0.09878
NOISE_SENSOR_SENSITIVITY = 0.3

# sample rate in samples per channel per second.
RATE = 1000

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
        self.sample_rate = None
        self.timestamp = timestamp
        
        self.set_channel()
        self.init_setting()
        self.get_csv_writer()
    
    def set_channel(self):
        # Default Value: 1.0 volts per unit(V/unit)
        # Set sesitivity for V/Unit to mV/Unit
        self.configure_channel(LOW_CHANNEL, VIBRATION_SENSOR_SENSITIVITY) # mV/g
        self.configure_channel(HIGH_CHANNEL, NOISE_SENSOR_SENSITIVITY) # mV/Pa

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
        self.sample_rate = self.ai_device.a_in_scan(LOW_CHANNEL, HIGH_CHANNEL, AiInputMode.SINGLE_ENDED, Range.BIP10VOLTS, samples_per_channel, RATE, ScanOption.CONTINUOUS, AInScanFlag.DEFAULT, self.data)

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
            frame = 0
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
                    sys.stdout.write(f'\nElapsed Time = {elapsed_time:.6f} | Frame = {frame} | Actual scan rate = {self.sample_rate:.6f} | Current index = {index} | Scan count = {scan_count} | Total count = {total_count}')
                    frame += 1

                if not MAX_MEASUREMENT_SECOND == 0:
                    if elapsed_time >= MAX_MEASUREMENT_SECOND:
                        sys.stdout.writelines(f'\nMeasurement stopped after {elapsed_time:.2f} seconds\n')
                        break
                else:
                    # if MAX_MEASUREMENT_SECOND == 0 -> Set continous -> WIP
                    pass

                next_sleep_time += 1.0 / RATE
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
        self.max_freq = 1000
        self.reference_pressure = 20e-6  # 20 µPa in Pascals
        self.epsilon = 1e-12  # For error: divide by zero encountered in log10
        
        self.limited_frequencies, self.limited_magnitude = self._calculate_fft()
        self.actual_max_freq = round(self.limited_frequencies[-1], 1)
        self.db_spl_data = self._calculate_db_spl(self.data['channel_1_value'])
        self.db_spl_magnitude = self._calculate_db_spl(self.limited_magnitude)

    def _calculate_fft(self):
        sampling_rate = 1 / np.mean(np.diff(self.data['seconds']))
        n = len(self.data['channel_1_value'])
        fft_result = np.fft.fft(self.data['channel_1_value'])
        frequencies = np.fft.fftfreq(n, d=1/sampling_rate)

        positive_frequencies = frequencies[:n // 2]
        fft_magnitude = np.abs(fft_result)[:n // 2]

        indices = positive_frequencies <= self.max_freq
        return positive_frequencies[indices], fft_magnitude[indices]
        
    def _calculate_db_spl(self, value):
        return 20 * np.log10(np.abs(value) / self.reference_pressure + self.epsilon)
    
    def _plot(self, x, y, title, xlabel, ylabel, file_suffix, color='blue', label=None):
        plt.figure(figsize=(12, 6))
        plt.plot(x, y, color=color, label=label)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        if label:
            plt.legend()
        plt.grid(True)

        figure_name = f'{os.path.join(OUTPUT_PATH, self.timestamp)}/{self.timestamp}_{file_suffix}.png'
        plt.savefig(figure_name)
    
    def plot_sensor_data(self):
        self._plot(
            self.data['seconds'], 
            self.data['channel_1_value'], 
            'Sensor Value Over Time', 
            'Time (Seconds)', 
            'Sensor Value (mV)]', 
            'output_data'
        )

    def plot_fft(self):
        self._plot(
            self.limited_frequencies, 
            self.limited_magnitude, 
            f'Frequency Spectrum (0-{self.actual_max_freq} Hz)', 
            'Frequency (Hz)', 
            'Magnitude', 
            'output_noise_fft', 
            color='red'
        )

    def plot_db_spl(self):
        self._plot(
            self.data['seconds'], 
            self.db_spl_data, 
            'Noise Sensor Data Over Time (dB SPL)', 
            'Time (Seconds)', 
            'dB SPL', 
            'output_noise_db_spl', 
            label='Noise Sensor Data in dB SPL'
        )

    def plot_fft_db_spl(self):
        self._plot(
            self.limited_frequencies, 
            self.db_spl_magnitude, 
            f'Noise Sensor Data in Frequency Domain (dB SPL, 0-{self.actual_max_freq} Hz)', 
            'Frequency (Hz)', 
            'dB SPL Magnitude', 
            'output_noise_fft_db_spl', 
            label='Noise Sensor Data in dB SPL'
        )

    def run(self):
        self.plot_sensor_data()
        self.plot_fft()
        self.plot_db_spl()
        self.plot_fft_db_spl()

if __name__ == '__main__':
    timestamp = get_timestamp()
    logger = DataLogger(timestamp)
    logger.start_scan()
    noise_vis = DataVisualizationCSV(timestamp)
    noise_vis.run()