#!/usr/bin/env python
# coding: utf-8

from csv import reader
import pandas as pd
import numpy as np
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

def main():
	# Iterating over the input file and appending each row to the items list.
	items = []
	with open('market_items.csv', 'r') as csvfile:
		csv_reader = reader(csvfile)
		for row in csv_reader:
			items.append(row)
	
	# Getting a peak into the items list.
	print("### items[0:5]:")
	print(items[0:5])
	print("\n\n")
	
	# Instantiating a TransactionEncoder object.
	encoder = TransactionEncoder()
	
	# Calling the fit method to extract the unique labels in the items list
	# immediately followed by the transform method to transform to boolean numpy array.
	transactions = encoder.fit(items).transform(items)
	print("### transactions")
	print(transactions)
	print("\n\n")

	# Creating a dataframe from the transactions using the same column labels.
	itemsets = pd.DataFrame(transactions, columns = encoder.columns_)
	
	# Previewing the dataframe.
	print("### itemsets.head()")
	print(itemsets.head())
	print("\n\n")

	# Seeing the summary of the structure of the dataframe.
	print("### itemsets.info()")
	print(itemsets.info())
	print("\n\n")

	# MAIN CALL: Apriori algorithm
	# Finding the frequent itemsets using the Apriori algo.
	# The minimum support is set to 0.01. That can be interpreted to focusing on itemsets that 
	# occur at least once per day. There are 306 days in the dataset and 21,664 
	# transactions. So the minimum support value is 1 * 306 / 21664 ~ 0.01.
	# We also set use_colnames to True, to have the names of items instead of indices.
	frequent_itemsets = apriori(itemsets, min_support = 0.01, use_colnames = True)
	print("### frequent_itemsets")
	print(frequent_itemsets)
	print("\n\n")

	# Getting the itemsets with highest support values.
	print("### frequent_itemsets.sort_values('support', ascending = False)")
	print(frequent_itemsets.sort_values('support', ascending = False))
	print("\n\n")

	# We can tighten our attention to a specific length range for itemsets.
	# For example: length > 2
	length = frequent_itemsets['itemsets'].str.len()
	rows = length > 2
	print("### frequent_itemsets[rows]")
	print(frequent_itemsets[rows])
	print("\n\n")

	# Looking at the bigger picture of the distribution of the support values in the dataframe.
	print("### frequent_itemsets.groupby(length)['support'].describe()")
	print(frequent_itemsets.groupby(length)['support'].describe())
	print("\n\n")

	# ASSOCIATION RULES
	# Focusing on rules with (e.g.,) minimum confidence of 0.5
	rules = association_rules(frequent_itemsets, metric = 'confidence', min_threshold = 0.5)
	print("### rules")
	print(rules)
	print("\n\n")

	# Looking at the created rules, and getting to understand the distribution based on 
	# different metrics, such as lift, leverage, and conviction.
	# We can use different metrics with different criteria to find suggestions.
	print("### rules.describe()")
	print(rules.describe())
	print("\n\n")

	print("### rules.sort_values('lift', ascending = False).head()")
	print(rules.sort_values('lift', ascending = False).head())
	print("\n\n")

	print("### rules.sort_values('leverage', ascending = False).head()")
	print(rules.sort_values('leverage', ascending = False).head())
	print("\n\n")

	print("### rules.sort_values('conviction', ascending = False).head()")
	print(rules.sort_values('conviction', ascending = False).head())
	print("\n\n")

	# The Zhang metric is defined as a method at the bottom of the code.
	rules['zhang'] = zhang_metric(rules)
	print("### rules")
	print(rules)
	print("\n\n")

	print("### rules.sort_values('zhang', ascending = False).head()")
	print(rules.sort_values('zhang', ascending = False).head())
	print("\n\n")

	print("### rules.sort_values('zhang').head()")
	print(rules.sort_values('zhang').head())
	print("\n\n")


def zhang_metric(rules):
	sup = rules['support'].copy()
	sup_a = rules['antecedent support'].copy()
	sup_b = rules['consequent support'].copy()
	num = sup - sup_a * sup_b
	denom = np.max((sup * (1 - sup_a).values, sup_a * (sup_b - sup).values), axis = 0)
	return num / denom
	
	
if __name__ == '__main__':
    main()

