# import the necessary packages
from pyimagesearch.utils.ranked import rank5_accuracy
import argparse
import pickle
import h5py

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--db", required=True, help="path HDF5 file")