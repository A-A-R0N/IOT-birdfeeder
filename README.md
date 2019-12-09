# IOT-birdfeeder
Attempts to identify birds vs squirrels using computer vision and weight.

## Training Setup
Follow steps at https://www.tensorflow.org/install/gpu for installing CUDA Toolkit and cuDNN SDK.

Download image data using https://github.com/EscVM/OIDv4_ToolKit selecting squirrel and bird classes.

Install requirements
Create virtual environment - https://programwithus.com/learn-to-code/Pip-and-virtualenv-on-Windows/
```
pip install -r training_requirements.txt
```
## Rasperry Pi Setup

### Linux packages Installation
```
sudo apt-get install libjasper-dev  
sudo apt-get install libqt4-test  
sudo apt install libqtgui4  
sudo apt-get install libatlas-base-dev  
sudo apt-get install libatomics-ops-dev  
sudo apt-get install libhdf5-dev  
```

### Virtual Environment Config
From project directory:  
```
python3 -m venv venv  
source venv/bin/activate  
```
This should put you in venv shell for pip install:  
```
pip3 install -r raspberrypi-requirements  
```

### Launching the code
You will need to locate the libatomic library installed in the last step using the following command:  
```
sudo find / -type f -name '*atom*.so*'  
```
The result will have to be added to the startup command as the LD_PRELOAD value. On my machine this was:  
```
LD_PRELOAD=/usr/lib/arm-linux-gnueabihf/libatomic.so.1.2.0 python3 bs_camera_test.py  
```
You will still need to run this from the venv shell: 
```
source venv/bin/activate  
```
