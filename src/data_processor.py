# python3 linux
# Project:  Keshav
# Filename: gtrends_downloader.py
# Author:   Raymond G. Nucuta (rnucuta@gmail.com)

"""
gtrends_downloader: Script that downloads weekly and monthly data on a specific disease in a geographical area.
					Saves this data as a csv to be used by gtrends_neutralizer.py
"""


from pytrends.request import TrendReq
import argparse
import pandas as pd
import os
from datetime import datetime 
from collections import defaultdict
from tqdm import tqdm


#default command: python3 data_processor.py --location US-AZ-753

parser = argparse.ArgumentParser()
parser.add_argument("--location", default='US-AZ-753', help="Location to be training on. Default is Pheonix, AZ.")
parser.add_argument("--start_time", default=2010, help="Time to start gathering training data from. Default is 2010.")
parser.add_argument("--disease", default='valley fever', help="Disease to gather relevant training data about. Default is valley fever.")
args = parser.parse_args()


#get arrays of time values for every month and each year since args.start_time
current_time = datetime.now()
months = []
for i in range(current_time.year+1-args.start_time):
	if args.start_time+i!=current_time.year:
		for j in range(12):
			if j+1<10:
				months.append('{}-0{}-01'.format(args.start_time+i, j+1))
			else:
				months.append('{}-{}-01'.format(args.start_time+i, j+1))
	else:
		for j in range(current_time.month):
			if j+1<10:
				months.append('{}-0{}-01'.format(args.start_time+i, j+1))
			else:
				months.append('{}-{}-01'.format(args.start_time+i, j+1))


years = []
for i in range(current_time.year+1-args.start_time):
	years.append('{}-01-01'.format(i+args.start_time))


#set up pytrends
pytrends = TrendReq(hl='en-US', tz=360)
#Use proxies because blocked by Google rate limit
# pytrends = TrendReq(hl='en-US', tz=360, timeout=(10,25), proxies=['https://34.203.233.13:80',], retries=2, backoff_factor=0.1)


#get up to 151 queries related to lyme disease
def get_related_queries(disease):
	print("Getting queries...", end=" ")
	all_queries =[disease]
	pytrends.build_payload([disease], cat=0, timeframe='today 5-y', geo=args.location, gprop='')
	for i, j in pytrends.related_queries()[disease]['top'].iterrows():
		all_queries.append(j['query'])
		pytrends.build_payload([j['query']], cat=0, timeframe='today 5-y', geo=args.location, gprop='')
		count = 0
		try:
			for k, l in pytrends.related_queries()[j['query']]['top'].iterrows():
				all_queries.append(l['query'])
				count+=1
				if count==5:
					break
		except AttributeError:
			pass
	#remove duplicates
	all_queries = list(dict.fromkeys(all_queries))
	print('{} queries'.format(len(all_queries)))
	return all_queries


def get_weekly_data(queries_list):
	print("Getting weekly data...")
	data_for_each_query = defaultdict(list)
	query_count=0
	for query in tqdm(queries_list):
		for i in range(len(years)-1):
			pytrends.build_payload([query], cat=0, timeframe='{} {}'.format(years[i], years[i+1]), geo=args.location, gprop='')
			query_data = pytrends.interest_over_time()
			data_for_each_query[query].append(query_data)
		# print(query_count)
		query_count+=1
	print("Done!")
	# print(data_for_each_query)
	return data_for_each_query

def get_monthly_data(queries_list):
	print("Getting monthly data...")
	data_for_each_query = defaultdict(list)
	query_count=0
	for query in tqdm(queries_list):
		pytrends.build_payload([query], cat=0, timeframe='all', geo=args.location, gprop='')
		query_data = pytrends.interest_over_time()
		data_for_each_query[query].append(query_data)
		# print(query_count)
		query_count+=1
	print("Done!")
	# print(data_for_each_query)
	return data_for_each_query

def save_data(data_dict, time_frame):
	print("Saving Data...")
	dates_index = []
	for data in data_dict[args.disease]:
		for date in data[args.disease].keys():
			dates_index.append(str(date))

	data_dict2 = data_dict.copy()
	for query in data_dict:
		values = []
		try: 
			for data in data_dict[query]:
				for value in data[query]:
					values.append(value)
			data_dict2[query]=values.copy()
		except KeyError:
			del data_dict2[query]
			print("Key error while saving: {}".format(query))

	df = pd.DataFrame(data_dict2, index=dates_index) 
	df.to_csv('../dumps/{}_{}.csv'.format(args.disease, time_frame))
	# print(data_dict2)
	return data_dict2

def transform_weekly_data():
	pass

save_data(get_weekly_data(get_related_queries(args.disease)), "weeklyData")
# save_data(get_monthly_data(get_related_queries(args.disease)), "monthlyData")