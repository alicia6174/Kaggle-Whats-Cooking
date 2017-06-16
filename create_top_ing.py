#!/usr/bin/python
import numpy as np
import json
import operator
import csv


with open("./train.json") as f:
    data = json.load(f)
    cus = np.array([], dtype=object)
    ing = np.array([], dtype=object)
    num_cus = 0
    num_ing = 0

    ## Construct cus_ing data
    cus_ing = {}
    for i in range(0, len(data)):
       buf_cus = data[i]["cuisine"]
       if ( buf_cus not in cus_ing.keys() ):
           cus_ing[buf_cus] = {}
       ith_ing = data[i]["ingredients"]
       for j in range(0, len(ith_ing)):
           buf_ing = ith_ing[j].encode('utf-8')
           if ( buf_ing not in cus_ing[buf_cus].keys() ):
              cus_ing[buf_cus][buf_ing] = 1
           else:
              cus_ing[buf_cus][buf_ing] =\
              cus_ing[buf_cus][buf_ing] + 1

    ## Find the top ingredients for each cuisine
    num_top = 200
    sorted_cus_ing = cus_ing
    for k in range(0, len(cus_ing)):
       buf_cus = cus_ing.keys()[k]
       sorted_cus_ing[buf_cus] =\
               sorted(cus_ing[buf_cus].items(),\
               key=operator.itemgetter(1),\
               reverse=True)[0:num_top]
    top_ing = np.array([], dtype=object)
    num_top_ing = 0
    for k in range(0, len(cus_ing)):
       buf_cus = sorted_cus_ing[cus_ing.keys()[k]]
       for l in range(0, num_top):
          # print "err:" + buf_cus[l][0];
          if ( buf_cus[l][0].encode('utf-8') not in top_ing ):
             top_ing = np.insert(top_ing, num_top_ing, buf_cus[l][0].encode('utf-8'))
             num_top_ing = num_top_ing + 1

## Write top ingredients as a csv file
with open('ing_top.csv', 'w') as csvfile:
    fieldnames = top_ing
    fieldnames = np.insert(fieldnames, 0, 'cuisine')
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for k in range(0,len(cus_ing)):
        wline = {};
        wline['cuisine'] = cus_ing.keys()[k]
        buf_cus = sorted_cus_ing[cus_ing.keys()[k]]
        for l in range(1, len(fieldnames)):
           for m in range(0, num_top):
               if (buf_cus[m][0] == fieldnames[l]):
                   wline[fieldnames[l]] = buf_cus[m][1]
        writer.writerow(wline);
