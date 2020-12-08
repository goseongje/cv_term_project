import os
import argparse
import numpy as np
import tensorflow as tf
import keras

import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser(description='Training of Network using training data')
    parser.add_argument('--gpu', type=str, default="0,1", help='gpu id')    
    args = parser.parse_args()    