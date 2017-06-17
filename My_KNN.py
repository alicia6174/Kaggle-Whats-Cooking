#!/usr/bin/python
import numpy as np
import json
import operator
import csv

K = 29
ret_cus = np.array([], dtype=object)

# with open("CNN.json") as f:
with open("train.json") as f:
    CNN = json.load(f)

with open("test.json") as g:
    test = json.load(g)
    id_cus = np.zeros((len(test),), dtype=np.int)
    for i in range(0, len(test)):
    # for i in range(0, 3):
        print 'i='+ str(i)
        can_cus = {\
        'greek':0, 'southern_us':0, 'filipino':0, 'indian':0, 'jamaican':0,\
        'spanish':0, 'italian':0, 'mexican':0, 'chinese':0, 'british':0,\
        'thai':0, 'vietnamese':0, 'cajun_creole':0, 'brazilian':0, 'french':0,\
        'japanese':0, 'irish':0, 'korean':0, 'moroccan':0, 'russian':0}
        dist = np.zeros((len(CNN),), dtype=np.int)
        for j in range(0, len(CNN)):
        # for j in range(0, 3):
            dist[j] = len(set(test[i]['ingredients']) &\
                             set(CNN[j]['ingredients']))
        indx = np.argsort(-dist)
        for k in range(0, K):
            tmp_cus = CNN[indx[k]]['cuisine']
            can_cus[tmp_cus] = can_cus[tmp_cus] + 1
        top_cus = sorted(can_cus.iteritems(),\
                  key=lambda (k,v): (v,k), reverse=True)[0][0]
        id_cus[i] = test[i]['id']
        ret_cus = np.insert(ret_cus, i, top_cus)

print len(ret_cus)
with open('submission.csv', 'w') as csvfile:
    fieldnames = np.array([], dtype=object)
    fieldnames = np.insert(fieldnames, 0, 'id')
    fieldnames = np.insert(fieldnames, 1, 'cuisine')
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for i in range(0, len(test)):
    # for i in range(0, 3):
        wline = {};
        wline['id'] = id_cus[i]
        wline['cuisine'] = ret_cus[i]
        writer.writerow(wline);
    # for i in range(0, len(data)): # to make the same order of id as sample
    #     wline = {};
    #     wline['id'] = id_smp[i]
    #     for j in range(0, len(data)):
    #         if ( id_cus[j] == id_smp[i] ):
    #             wline['cuisine'] = ret_cus[j]
    #     writer.writerow(wline);

#CNN.json
#K=1  => 0.53691
#K=3  => 0.55471
#K=5  => 0.61394
#K=7  => 0.63284
#K=9  => 0.63988
#K=11 => 0.64853
#K=13 => 0.65296
#K=15 => 0.65718
#K=17 => 0.65426
#K=19 => 0.65698
#K=21 => 0.65788
#K=23 => 0.65879
#K=25 => 0.65788
#K=27 =>
#K=105 =>
#K=205 =>
#K=305 =>
#K=405 =>
#K=505 =>
#K=605 =>
#K=705 =>
#K=805 =>
#K=905 =>
#K=1005 =>
#K=1105 =>
#K=1205 =>
#K=1305 =>
#K=1405 =>
#K=1501 => 0.46862

#train.json
#K=21 => 0.67659
#K=23 => 0.67508
#K=25 => 0.67548
#K=27 => 0.67347
#K=29 => 0.67267
#K=35 => 0.67086
#K=45 => 0.66653
#K=55 =>
#K=65 =>
#K=75 =>
#K=85 =>
#K=95 =>
#K=105 => 0.63868
#K=199 => 0.61283
#K=205 => 0.61223
#K=305 => 0.59161
#K=405 => 0.57763
#K=505 => 0.56879
#K=605 => 0.55893
#K=705 => 0.55109
#K=805 => 0.54314
#K=905 => 0.53842
#K=1005 => 0.53439
#K=1105 =>
#K=1205 =>
#K=1305 =>
#K=1405 =>
#K=2005 => 0.50080
#K=3005 => 0.48270
