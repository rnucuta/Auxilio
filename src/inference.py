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

#=IF(C2<66, 0, IF(C2>156.5, 2, 1))

import argparse
from pytrends.request import TrendReq
import pandas as pd
from datetime import datetime
from datetime import date
from collections import defaultdict
import joblib
import os
from tqdm import tqdm

class color:
   PURPLE = '\033[95m'
   CYAN = '\033[96m'
   DARKCYAN = '\033[36m'
   BLUE = '\033[94m'
   GREEN = '\033[92m'
   YELLOW = '\033[93m'
   RED = '\033[91m'
   BOLD = '\033[1m'
   UNDERLINE = '\033[4m'
   END = '\033[0m'

def load_model(model_file):
    return joblib.load(os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', 'models',model_file)))

def load_data(csv_name):
    df = pd.read_csv('../dumps/'+csv_name)
    # for i in range(len(df.index)):
    #     temp_date=datetime.strptime(df.at[i,"Date"], '%m/%d/%Y %M:%S')
    #     df.at[i,'Date']=int(temp_date.month)
    return df

def get_string_dates():
    year1=datetime.now().year-1
    year2=datetime.now().year
    final1=str(year1)
    final2=str(year2)
    if datetime.now().month<10:
        final1+="-0"+str(datetime.now().month)
        final2+="-0"+str(datetime.now().month)
    else:
        final1+="-"+str(datetime.now().month)
        final2+="-"+str(datetime.now().month)
    if datetime.now().day<10:
        final1+="-0"+str(datetime.now().day)
        final2+="-0"+str(datetime.now().day)
    else:
        final1+="-"+str(datetime.now().day)
        final2+="-"+str(datetime.now().day)
    return final1, final2

def get_this_weeks_data(training_data, location):
    pytrends = TrendReq(hl='en-US', tz=360)
    cols = [col for col in training_data.columns if col not in ['Date','DiseaseIncidence', 'AdjustedDiseaseIncidence', 'LOW', 'MEDIUM']]
    month=datetime.now().month
    year1, year2 = get_string_dates()

    data_for_each_query = defaultdict(list)
    data_for_each_query['Date'].append(month)

    for query in tqdm(cols):
        pytrends.build_payload([query], cat=0, timeframe='{} {}'.format(year1, year2), geo=location, gprop='')
        query_data = pytrends.interest_over_time()
        data_for_each_query[query].append(query_data.at[query_data.index[len(query_data.index)-1], query])

    return pd.DataFrame.from_dict(data_for_each_query)

def infer(loaded_model, data):
    output=loaded_model.predict(data)[0]
    print()
    print()
    print("Disease incidence for week {}/52: ".format(color.YELLOW+str(date(datetime.now().year, datetime.now().month,datetime.now().day).isocalendar()[1]))+color.END, end="")
    # print(output)
    if output==0:
        print(color.PURPLE+"LOW INCIDENCE"+color.END)
        return "LOW"
    if output==1:
        print(color.PURPLE+"MEDIUM INCIDENCE"+color.END)
        return "MEDIUM"
    elif output==2:
        print(color.PURPLE+"HIGH INCIDENCE"+color.END)
        return "HIGH"
    else:
        print(color.RED+"Inference gave error?!"+color.END)

if __name__ == '__main__':
    # Parse command line arguments.

    #default command: python3 inference.py --model_file "KNeighbors acc 0.551.sav" --disease_freq "valley fever_weeklyData - Copy.csv"

    parser = argparse.ArgumentParser(description=__doc__)
    # Data files/directories.
    parser.add_argument('--model_file', required=True, \
                        help='name of .sav file of a trained model that is in /models')
    parser.add_argument('--disease_freq', required=True, \
                        help='name of .csv file with weekly trends/incidence data that is in /dumps')
    parser.add_argument("--location", default='US-AZ', help="Location to be training on. Default is AZ.")
    args = parser.parse_args()

    load_model=load_model(args.model_file)
    loaded_df=load_data(args.disease_freq)

    current_data=get_this_weeks_data(loaded_df, args.location)
    print(current_data.head())

    inference_output=infer(load_model, current_data)