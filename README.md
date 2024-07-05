# SAIL Parser
Real time data parsing and plotting for VortEx. Made for the Space and Atmospheric Instrumentation Lab.

## Setup
python >= 3.10  
Install the required libraries from the `requirements.txt` file through the command:  
`$ pip install -r requirements.txt`

Or download the libraries manually:   
`numpy, vispy, openpyxl, PyQt5, scipy, pymap3d`

Run `main.py` from the source folder

## Using the parser

![image](https://github.com/yashjain128/vortex_parser/assets/54511272/4404e72c-0330-4901-8546-bb172b7463a0)

Select an format file and map file with the dropdown or file explorer
Files in the lib folder will be available in the dropdown

Either select Read File and choose a file
Or select UDP and specify the port info

## Project Organization
The program is split into a files: `main.py` , `gui.py` , `plotting.py`

`main.py` is the runnable file that handles all of the other files  
`plotting.py` contains everything related to plotting through matplotlib
`gui.py` has the pyqt application.

To simulate a connection: `sendUDP.py`  

All of the udp data files are in the lib folder

## Pending changes

These will be implemented in the future
- Live UDP parsing

##
Contributed by Yash Jain
