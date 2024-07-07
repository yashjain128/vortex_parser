# SAIL Parser
Real time data parsing and plotting for VortEx. Made for the Space and Atmospheric Instrumentation Lab.

##
### Setup
Requires python >= 3.10
Install the required libraries from the `requirements.txt` file through the command:  
`$ pip install -r requirements.txt`

Or download the libraries manually:   
`numpy, vispy, openpyxl, PyQt5, scipy, pymap3d`

Run `main.py` from the source folder

##
### Using the parser

![image](https://github.com/yashjain128/vortex_parser/assets/54511272/4404e72c-0330-4901-8546-bb172b7463a0)

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
- Testing

##
Contributed by Yash Jain
jainy7002@gmail.com
