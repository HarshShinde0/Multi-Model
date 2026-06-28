import os
import sys

# Add the src directory to the python path so tests can import multimodel_app
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
