# VortEx Parser
Real time data parsing and plotting for VortEx. Made for the Space and Atmospheric Instrumentation Lab at ERAU.

##
### Setup
Requires python >= 3.10

Install the required libraries from the `requirements.txt` file through the command:  
`$ pip install -r requirements.txt`

Or download the libraries manually: `numpy, vispy, openpyxl, PyQt5, scipy, pymap3d`

##
### Using the parser

Run `main.py` from the `src` folder

##
### Project Organization
The program is split into a files: `main.py` , `gui.py` , `plotting.py`

`main.py` is the runnable file that handles all of the other files  
`plotting.py` contains everything related to plotting through matplotlib
`gui.py` has the pyqt application

The lib folder is where the udp data files, .mat map files, and .xlsx format files are located

File selection will default to the lib folder

##
### Pending changes

These will be implemented in the future
- Live UDP parsing
- Better excel format
- Faster Parsing
- Testing

##
Contributed by Yash Jain

jainy7002@gmail.com
