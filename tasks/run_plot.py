import json
import pandas as pd
import numpy as np
import pickle
import os
from os import listdir
from os.path import isfile, join
import sys
import fire
import operator

# User generated python files
import plot

def start():
	df_fda = pd.read_csv("./data/clue_to_fda.csv", sep =",", header=None)
	df_info = pd.read_csv("./data/drug_info_2.0.csv", sep =",")

	with open("./data/99_our_v0.6_py2.pkl", 'rb') as f:
	    df_embed = pickle.load(f)

	#task0(df_fda, df_info, df_embed)
	#task1(df_fda, df_info, df_embed)
	#task2(df_fda, df_info, df_embed)
	#task3(df_fda, df_info, df_embed)
	task4(df_fda, df_info, df_embed)


def count_moa():
	dict_moa_count = {}

	sorted_x = sorted(dict_moa_count.items(), key=lambda x: x[1], reverse=True)
	top10_list = []

	for i in sorted_x[:10]:
		top10_list.append(i[0])

	print(sorted_x)

# original
def task0(df_fda, df_info, df_embed):
	dict_id_embed = {}
	dict_id_moa = {}
	df_fda = df_fda[df_fda[3] == 'fda']
	fda_list = df_fda[1].unique()

	for _id in fda_list:
		try:
			moa = df_info[df_info['pert_id'] == _id]['moa'].values[0].split("|")[0]
		except AttributeError:
			moa = "N/A"

		iname = df_info[df_info['pert_id'] == _id]['pert_iname'].values[0]
		embed = df_embed[_id][0]


		dict_id_embed[_id] = np.array(embed).reshape(1, -1)
		dict_id_moa[_id] = moa

	print(len(dict_id_embed))
	dict_id_embed_tsne = plot.load_TSNE(dict_id_embed, dim=2)
	plot.plot_category(dict_id_embed, dict_id_embed_tsne, "./data/plottings/task0", dict_id_moa, True)

def task1(df_fda, df_info, df_embed):
	dict_id_embed = {}
	dict_id_moa = {}
	df_fda = df_fda[df_fda[3] == 'fda']
	fda_list = df_fda[1].unique()

	top10_list = ['Dopamine receptor antagonist', 'Cyclooxygenase inhibitor', 'Histamine receptor antagonist', 'Adrenergic receptor agonist', 'Adrenergic receptor antagonist', 'Bacterial cell wall synthesis inhibitor',
							'Acetylcholine receptor antagonist', 'Glucocorticoid receptor agonist', 'Serotonin receptor antagonist', 'Sodium channel blocker']

	for _id in fda_list:
		try:
			moa = df_info[df_info['pert_id'] == _id]['moa'].values[0].split("|")[0]
		except AttributeError:
			moa = "N/A"
		iname = df_info[df_info['pert_id'] == _id]['pert_iname'].values[0]
		embed = df_embed[_id][0]

		_id_name = _id+"@"+iname

		if moa in top10_list:
			dict_id_embed[_id_name] = np.array(embed).reshape(1, -1)
			dict_id_moa[_id_name] = moa

	print(len(dict_id_embed))
	dict_id_embed_tsne = plot.load_TSNE(dict_id_embed, dim=2)
	plot.plot_category(dict_id_embed, dict_id_embed_tsne, "./data/plottings/task1", dict_id_moa, True)

def task2(df_fda, df_info, df_embed):
	dict_id_embed = {}
	dict_id_moa = {}
	df_fda = df_fda[df_fda[3] == 'fda']
	fda_list = df_fda[1].unique()

	for _id in df_embed:
		try:
			moa = df_info[df_info['pert_id'] == _id]['moa'].values[0].split("|")[0]
		except AttributeError:
			moa = "N/A"
		iname = df_info[df_info['pert_id'] == _id]['pert_iname'].values[0]
		embed = df_embed[_id][0]

		_id_name = _id+"@"+iname
		dict_id_embed[_id_name] = np.array(embed).reshape(1, -1)

		if _id in fda_list:
			dict_id_moa[_id_name] = moa
		else:
			dict_id_moa[_id_name] = "None FDA"

	print(len(dict_id_embed))
	dict_id_embed_tsne = plot.load_TSNE(dict_id_embed, dim=2)
	plot.plot_category(dict_id_embed, dict_id_embed_tsne, "./data/plottings/task2", dict_id_moa, True)

def task3(df_fda, df_info, df_embed):
	dict_id_embed = {}
	dict_id_moa = {}
	df_fda = df_fda[df_fda[3] == 'fda']
	fda_list = df_fda[1].unique()

	for _id in df_embed:
		try:
			moa = df_info[df_info['pert_id'] == _id]['moa'].values[0].split("|")[0]
		except AttributeError:
			moa = "N/A"
		iname = df_info[df_info['pert_id'] == _id]['pert_iname'].values[0]
		embed = df_embed[_id][0]

		_id_name = _id+"@"+iname
		dict_id_embed[_id_name] = np.array(embed).reshape(1, -1)

		if _id in fda_list:
			dict_id_moa[_id_name] = "FDA"
		else:
			dict_id_moa[_id_name] = "None FDA"

	print(len(dict_id_embed))
	dict_id_embed_tsne = plot.load_TSNE(dict_id_embed, dim=2)
	plot.plot_category(dict_id_embed, dict_id_embed_tsne, "./data/plottings/task3", dict_id_moa, True)

def task4(df_fda, df_info, df_embed):
	dict_id_embed = {}
	dict_id_moa = {}
	df_fda = df_fda[df_fda[3] == 'fda']
	fda_list = df_fda[1].unique()

	for _id in df_embed:
		try:
			moa = df_info[df_info['pert_id'] == _id]['moa'].values[0].split("|")[0]
		except AttributeError:
			moa = "N/A"
		iname = df_info[df_info['pert_id'] == _id]['pert_iname'].values[0]
		embed = df_embed[_id][0]

		_id_name = _id+"@"+iname


		top10_list = ['Dopamine receptor antagonist', 'Cyclooxygenase inhibitor', 'Histamine receptor antagonist', 'Adrenergic receptor agonist', 'Adrenergic receptor antagonist', 'Bacterial cell wall synthesis inhibitor',
								'Acetylcholine receptor antagonist', 'Glucocorticoid receptor agonist', 'Serotonin receptor antagonist', 'Sodium channel blocker']


		if moa in top10_list:
			dict_id_embed[_id_name] = np.array(embed).reshape(1, -1)
			if _id in fda_list:
				dict_id_moa[_id_name] = moa
			else:
				dict_id_moa[_id_name] = "None FDA"

	print(len(dict_id_embed))
	dict_id_embed_tsne = plot.load_TSNE(dict_id_embed, dim=2)
	plot.plot_category(dict_id_embed, dict_id_embed_tsne, "./data/plottings/task4", dict_id_moa, True)

if __name__ == '__main__':
    fire.Fire()
