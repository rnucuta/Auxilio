# python3 linux
# Project:  Keshav
# Filename: gtrends_neutralizer.py
# Author:   Raymond G. Nucuta (rnucuta@gmail.com)

"""
gtrends_neutralizer: Script that neutralizes the downloaded weekly data 
					along existing 2004 to present monthly data, making the data
					actually all relevant to each other. Manual inputing of CDC data is required, and then
					next step is training.

					IMPORTANT: Determined that there is no need for this as it will simply train on a year to year basis
"""

import pandas as pd

monthly_dataframe = pd.read_csv("../dumps/valley fever_monthlyData.csv")
weekly_dataframe = pd.read_csv("../dumps/valley fever_weeklyData.csv")

def date_in_time(week, month):
	inside=false
	month_year = month[:4]
	week_year = week[:4]
	month_day = month[5:7]
	week_day = week[5:7]

	return inside

def get_monthly_values(date, monthly_data):
	for i in monthly_data.index:



def transform_weekly_data(weekly_data):
	for i in weekly_data.index:


print(monthly_data)