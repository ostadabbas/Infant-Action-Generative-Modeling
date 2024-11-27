License:
========
To learn about SMIL, please visit our website: http://s.fhg.de/smil

For comments or questions, please email us at: smil@iosb.fraunhofer.de

The Python code is adapted from the SMPL package, copyright 2015 Matthew Loper, Naureen Mahmood and
the Max Planck Gesellschaft.

System Requirements:
====================
Operating system: OSX, Linux

Python Dependencies:
- Numpy & Scipy  [http://www.scipy.org/scipylib/download.html]
- Chumpy 		 [https://github.com/mattloper/chumpy]
- OpenCV 		 [http://opencv.org/downloads.html] 


Getting Started:
================

1. Extract the Code:
--------------------
Extract the 'smil.zip' file to your home directory (or any other location you wish)


2. Set the PYTHONPATH:
----------------------
We need to update the PYTHONPATH environment variable so that the system knows how to find the SMIL code. Add the following lines to your ~/.bash_profile file (create it if it doesn't exist; Linux users might have ~/.bashrc file instead), replacing ~/smil with the location where you extracted the smil.zip file:

	SMIL_LOCATION=~/smil
	export PYTHONPATH=$PYTHONPATH:$SMIL_LOCATION


Open a new terminal window to check if the python path has been updated by typing the following:
>  echo $PYTHONPATH


3. Run the Hello World scripts:
-------------------------------
In the new Terminal window, navigate to the smil/smil_webuser/hello_world directory. You can run the hello world scripts now by typing the following:

> python hello_smil.py

OR 

> python render_smil.py



Note:
Both of these scripts will require the dependencies listed above. The scripts are provided as a sample to help you get started. 

