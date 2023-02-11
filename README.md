# Analyzing consumer behavior using Association Rules in Python.

The dataset is downloadable from [Kaggle](https://www.kaggle.com/datasets/aslanahmedov/market-basket-analysis/download?datasetVersionNumber=2).

## Dataset Descritpion

* File name: market.csv 
* Rows: 522,065 
* Attributes (7): 
	- BillNo: 6-digit number assigned to each transaction. (Nominal) 
	- Itemname: Product name. (Nominal) 
	- Quantity: The quantities of each product per transaction. (Nominal) 
	- Date: The day and time when each transaction was generated. (Numeric) 
	- Price: Product price. (Numeric) 
	- CustomerID: 5-digit number assigned to each customer. (Nominal) 
	- Country: Name of the country where each customer resides. (Nominal) 


## Methodology

Each row in the input file (market.csv) shows info for one transaction. A bill may consist of several transactions distributed in multiple rows. Therefore, we need to first aggregate rows based on the bill number (column 1). This could be done in Python, but I usually do these simple initial data manipulations in Bash. Bash has many powerful and easy to use tools to work with file rows/columns.

The following command simply turns the input file into a file with only 1 column of all items in the same bill, separated by comma:

```
cat market.csv | sed 's/,/./g' | sed 's/;/,/g' | cut -d ',' -f 1,2 | awk -F, '{arr[$1] = arr[$1] "," $2} END {for (i in arr) {print i,arr[i]}}' | sort -n | cut -d ',' -f 2- > market_items.csv
```

This command, first prints the content of the input file using cat. It pipes this content to two consecutive sed commands. The first one replaces the comma in the payment amount (the dataset is European) with a dot. The second sed, replaces the semicolons with comma, so the file becomes comma separated. The output is piped to a cut command which takes only the first (bill number) and second (item name) columns. Then it is passed to an awk command which concatenates  the second columns of all rows with the same first column, using an array. It then prints out the first column (bill number) and aggregated second column (all items in the bill separated by comma). Then it sorts the transactions based on bill numbers and passes it to another cut, to cut out the bill numbers. The output is one column of all items in the same bill, separated by comma. It is all written into 'market\_items.csv'.

After data preparation, the following are the list steps to take:

1. Iterating over the input file and appending each row to the items list.
2. Getting a peak into the items list and understanding it.
3. Instantiating a TransactionEncoder object.
4. Calling the fit method to extract the unique labels in the items list immediately followed by the transform method to transform to boolean numpy array.
5. Creating a dataframe from the transactions using the same column labels.
6. Previewing the dataframe, seeing the summary of its structure. 
7. MAIN CALL: Apriori algorithm: Finding the frequent itemsets using the Apriori algo. The minimum support is set to 0.003, since we want to focus on itemsets that occur at least 5 times per day. There are 374 days in the dataset and 21,664 transactions. So the minimum support value is 5 * 306 / 522064 = 0.003. We also set use_colnames to True, to have the names of items instead of indices.
8. Getting the itemsets with highest support values.
9. We can tighten our attention to a specific length range for itemsets. For example: length > 2
10. Looking at the bigger picture of the distribution of the support values in the dataframe.
11. ASSOCIATION RULES: Focusing on rules with (e.g.,) minimum confidence of 0.5
12. Looking at the created rules, and getting to understand the distribution based on different metrics, such as lift, leverage, and conviction. We can use different metrics with different criteria to find suggestions.
13. The Zhang metric can be defined as a method at the bottom of the code.

The .ipynb version looks at the top 5000 transactions and provides a peak into the results. The .py version of the code, however, is also available and looks at the whole data. I ran the .py version as a job on an HPC cluster with parallel computation and big memory space. The output is stored in 'market_association_rules.output'.