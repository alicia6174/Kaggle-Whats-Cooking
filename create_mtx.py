#!/usr/bin/python
import numpy as np
import json
import operator
import csv

g = open('ing.csv', 'r')
# flag = 0
for row in csv.reader(g):
    ing = row
    # print len(ing)
    # if flag == 1:
    #     # print len(row)
    #     break
    # ing = row[1:len(row)]#the 1st is 'cuisine'
    # flag = flag + 1
g.close()

NUM_ING = len(ing)
writer = csv.writer(file('test_mtx.csv', 'w'), lineterminator="\n")
# fieldnames = np.array(range(1, NUM_ING+1), dtype=object)
# fieldnames = np.insert(fieldnames, NUM_ING, NUM_ING+1)
# fieldnames = np.insert(fieldnames, NUM_ING+1, 'cuisine')
# writer.writerow(fieldnames);

with open("test.json") as f:
    data = json.load(f)
    train_mtx = np.array([ np.zeros((len(ing),), dtype=object) ])#from idx=1
    # ing_len = np.array([], dtype=float)
    for i in range(0, len(data)):
    # for i in range(0, 3):
        print i
        ith_ing = data[i]["ingredients"]
        ith_data = np.zeros((len(ing),), dtype=object)
        for j in range(0, len(ith_ing)):
            buf_ing = ith_ing[j].encode('utf-8')
            if (buf_ing in ing):
                buf_idx = ing.index(buf_ing)
                ith_data[buf_idx] = 1
        train_mtx = np.insert(train_mtx, i+1, ith_data, 0)
        # ing_len = np.insert(ing_len, i, len(data[i]['ingredients']), 0)

# ing_len = (ing_len-np.min(ing_len))/(np.max(ing_len)-np.min(ing_len))

for i in range(0, len(data)):
# for i in range(0, 3):
    wline = train_mtx[i+1];
    # wline = np.insert(wline, NUM_ING, ing_len[i])
    # wline = np.insert(wline, NUM_ING+1, data[i]['cuisine'].encode('utf-8'))
    # wline = np.insert(wline, NUM_ING+1, '?'.encode('utf-8'))
    writer.writerow(wline);

