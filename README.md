# UL For Linux Python API Sample
Using DT9837A Data Logger   
## Analog Inputs
* ~~Ain0: HS13M131 Accelerometers~~ (WIP)
* Ain1: EMM-82S-CTC Microphone

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
### 2. Install Dependency
```bash
pip install -r requirements.txt
```
## How to Use
```bash
python app.py
```