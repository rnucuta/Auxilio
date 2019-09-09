# python3 linux
# Project:  Trendsy
# Filename: inference.py
# Author:   Raymond G. Nucuta (rnucuta@gmail.com)

"""
inference: Script that runs inference. Needs no inputted data other than trained model.
		   Will gather the current month's and week's data, and output a 3-tiered risk 
		   rating of a particular disease (HIGH, MEDIUM, LOW) and corresponding recomendations
		   based on the risk rating. 
"""

import argparse
from pytrends.request import TrendReq
import pandas as pd
from datetime import datetime
import joblib
import os

def load_model(model_file):
    return joblib.load(os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', 'models',model_file)))

if __name__ == '__main__':
    # Parse command line arguments.

    #default command: python3 inference.py --model_file "elastic_net acc 0.305.sav"

    parser = argparse.ArgumentParser(description=__doc__)
    # Data files/directories.
    parser.add_argument('--model_file', required=True, \
                        help='name of .sav file of a trained model that is in /models')
    args = parser.parse_args()

    load_model=load_model(args.model_file)