#!/usr/bin/python
import numpy as np
import json
import operator
import csv

NUM_ING = 2000

writer = csv.writer(file('train_weka_tol2000_n_pca.csv', 'w'), lineterminator="\n")
fieldnames = np.array(range(1, NUM_ING+1), dtype=object)
fieldnames = np.insert(fieldnames, NUM_ING, 'cuisine')
writer.writerow(fieldnames);
# print fieldnames

with open("train.json") as f:
    data = json.load(f)

g = open('train_pca_mtx_K2000_n.csv', 'r')
i = 0
# flag = 0
for row in csv.reader(g):
    print i
    # if (flag == 2):
    #     exit()
    row = map(float, row)
    row = [ round(elem, 2) for elem in row ]
    wline = np.asarray(row, dtype=object);
    wline = np.insert(wline, NUM_ING, data[i]['cuisine'].encode('utf-8'))
    # wline = np.insert(wline, NUM_ING, '?'.encode('utf-8'))
    # print wline
    i = i + 1
    writer.writerow(wline);
    # flag = flag + 1
g.close()

