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

if __name__ == '__main__':
    # Parse command line arguments.

    #default command: python3 inference.py --disease_freq "valley fever_weeklyData.csv"

    parser = argparse.ArgumentParser(description=__doc__)
    # Data files/directories.
    parser.add_argument('--model_file', required=True, \
                        help='name of .csv file with weekly trends/incidence data that is in /dumps')
    # parser.add_argument('--model_file', required=True, \
    #                     help='name of model file that will after training is completed')
    args = parser.parse_args()