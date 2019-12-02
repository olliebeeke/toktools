# toktools
Some Python-tools for retrieving, plotting experimental data etc. WIP. This only currently retrieves data from
JETTO, and plots profiles and makes a GS2 input file.

---------------------------------------------------------- INSTALLATION ----------------------------------------------------------

i) Install the SAL python module (e.g. via "pip install sal". or equivalent)

ii) Unzip the JET SAL Client to a directory of your choice, and navigate there.

iii) Navigate to jet-dataclases in this directory, and run "python setup.py install".

iv) Navigate to jet-client in this directory, and run "python setup.py install"

v) Navigate to these tools, and in tt_main.py, set your SAL username for remote authentication, if necessary.

vi) Also in tt_main.py, edit the path to the example input file to be consistent with where it is stored.

You should now be ready to run the script!
