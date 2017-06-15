&nbsp; <p>
&nbsp; <p>
# <center> Machine Learning Kaggle Project </center>
# <center> What's Cooking </center>
&nbsp; <p>
&nbsp; <p>
&nbsp; <p>
<center> <img src="./pictures/Fig1.png" width="50%" /> </center>
&nbsp; <p>
&nbsp; <p>
&nbsp; <p>
&nbsp; <p>
&nbsp; <p>
### <center> 101021801 Yu-Hsuan Guan  </center>
### <center> NTHU Department of Mathematics</center>
### <center> June 28, 2017 </center>

<div style="page-break-after: always;"></div>

## 1. Introduction
### The main problem
The main issue of this competition is to classify lists of ingredients into correct kinds of cuisines. There are only two given files ``train.json`` and ``test.json``. Each training instance is represented in this format of JSON.  

```python
{
    "id": 10259,
    "cuisine": "greek",
    "ingredients": [ "romaine lettuce", "black olives", "grape tomatoes", "garlic",
    "pepper", "purple onion", "seasoning", "garbanzo beans", "feta cheese crumbles" ]
}
```
And, each testing instance is represented in the format of JSON.

```python
{
	"id": 18009,
	"ingredients": [ "baking powder", "eggs", "all-purpose flour", 
	"raisins", "milk", "white sugar" ]
}
``` 
  
There are totally 20 kinds of cuisines.
<center>

| irish | mexican | chinese | filipino | vietnamese |
| :-----------:| :--------:| :----------:| :-------:| :---------:|
| **moroccan** | **brazilian** | **japanese** | **british** | **greek** |
| **indian** | **jamaican** | **french** | **spanish** | **russian** |
| **cajun_creole** | **thai** | **southern_us** | **korean** | **italian** |

</center>

After an initial step of statistics, some basic summaries about data are given below.
<center>

| Number of training instances | 39774  |
| :---: | :---: |
| **Number of testing instances** | **9944** |
| **Length of longest list of ingregients in training data** | **65** |
| **Length of shortest list of ingregients in training data** | **1** |
| **Total number of ingregients in training data** | **6714** |

</center>

The submission needs to be saved as a csv file of this form.
<center>

| id | cuisine |
| :---: | :---: |
| 18009 | italian |
| 35203 | chinese |
| ... | ... |

</center>

Obviously, this competition is a **supervised problem** of **multi-class classification**. There are several kinds of applicable classifiers.

* Transform to binary - OvR, OvO* Extend from binary - Naive Bayes, KNN, Decision trees, SVM, Neural networks, ELM* Hierarchical classification

### The main steps of learning
There are five major steps of machine learning in this project.

* **Data analysis** - We dealt with **data preprocessing** and decided the form of data representation in this step which will be metnioned in the beginnings of sections 2 and 3.
* **Visualization** - According to the above result, the data matrix derived from the training instances has size of 39774 x 6714. To cope with this matrix more efficiently, we need **dimension reduction** to compress the size of matrix without losing too many varieties of data. This step is the main difference between related work and the new method mentioned in this project which will be mentioned in the sections 2 and 3.
* **Modeling** - We chose **Weka** environment to create models. The process of converting data matrix to the valid file for Weka environment will be mentioned in the sections 2 and 3. The detailed steps of how to use Weka will be described in the appendix.
* **Tuning** - In the section 3, we defined a specific score to determine the number of principal components used to reduce the dimension. And then, we used the  **AutoWeka** tool to choose appropriate parameters needed in the candidate models. The detailed steps of how to use AutoWeka will be described in the appendix.
* **Evaluation** - In the section 4, we presented various quantitiest to evaluate different models. These evidences showed that our method works better than the old one. Finally, we gave the scores of our submissions on Kaggle site.

### Data preprocessing 
* delete special characters(appendix) 
* convert to unitcode(appendix) 
* change to 01-vectors 
* a matrix of size 39774 x 6714 (training) without missing values -> very large!

Detailed steps.

* prefixFilter			create train.json  → delete special characters
* :set ff=unix			unicode
* a.encode(‘utf-8’)		unicode

## 2. Related work
* Top ing. -> sketch the method
* Dimension reduction -> Top ing.
* Modeling -> Weka (detailed steps)
* Tuning -> How to choose num. of top ing.?
* Testing
* Coding: appendix
* Best: Top 1000, S=0.57 (§4)
* We've tried Top 200 ing. + ing_len (normalized) -> Not good enough.
* The file size 81M vs. 187M due to the float type of PCA data.
Top 1000 ing. + ing_len (normalized).* create_top_ing.py		create ing\_top200.csv* create_mtx.py			create train\_weka\_top200\_len.csv (81M)* ./weka-csv-arff.pl < ./train\_weka\_top200\_len.csv > ./train\_weka\_top200\_len.arff
* Weka (Percen66%)* Naïve Bayes					63.3365 %
* IBk, k=1501					31.7533 % Take k=1501 for some reason
* J48							64.2387 %
* SMO							Out of memory!
* RandomTree					45.7739 % 
* RandomForest				Out of memory!
* MultiClassClassifier(OvR)Out of memory!
* MultiClassClassifier(OvO)Out of memory!* Testing???

## 3. New methods
* PCA + SMO -> sketch the method
* Dimension reduction -> PCA, linear unsupervised reduction
* Modeling -> Auto Weka + Weka (detailed steps)
* Tuning -> 1. How to choose num. of PCs? Score.pdf 2. How to choose parameters of SMo? AutoWeka
* Testing
* Coding: appendix
* Best: PCs 1000, S=0.66020 (§4)
* We've tried PCA 2000 (normalized) -> Out of memory while using Weka.

PCA 1000 (normalized).

* do\_pca.cpp			Have checked that eigenvectors are o.n.* divide\_into\_vec\_val.pl	* create\_pca_mtx.m			create train\_pca_mtx\_K1000\_n.csv (normalized)* create\_weka.py				create train\_weka\_tol1000\_n\_pca.csv (187M by round)* ./weka-csv-arff.pl < ./train\_weka\_tol1000\_n\_pca.csv > ./train\_weka\_tol1000\_n\_pca.arff
* Weka (Percen66%)
* Naïve Bayes					3x.xxx %
* IBk, k=1501					30.8364 % Take k=1501 for some reason
* J48							40.0281 %
* SMO							73.2382 %
* RandomTree					23.8335 %
* RandomForest				
* MultiClassClassifier(OvR)66.2797 %
* MultiClassClassifier(OvO)
* Weka							create train\_weka\_tol1000\_n\_pca\_SMO.model* create\_mtx.py				create test\_mtx.csv* create\_pca\_mtx.m			create test\_pca\_mtx\_K1000\_n.csv* create\_weka.py				create test\_weka\_tol1000\_n\_pca.csv (48.7M)* ./weka-csv-arff.pl < ./test\_weka\_tol1000\_n\_pca.csv > ./test\_weka\_tol1000\_n\_pca.arff* change the last attribute to cuisines* Weka							create test\_weka\_tol1000\_n\_pca\_SMO.txt 
* ./weka-to-kaggle.pl < ./ test\_weka\_tol1000\_n\_pca\_SMO.txt > ./test\_weka\_tol1000\_n\_pca\_SMO\_sub.csv
* Kaggle score: 0.66020
## 4. Comparison results
### Evaluation
* Put some of these results in the appendix since the matrix of results are too large!
* The runnung time is meaningless here since we used three devices to create all of the results simultaneously for convenience. 
* The devices are: MacBook Air, Fedora, Google
* Correctness / Accuracy / Error rate
* K-fold cross-validation
* Confusion matrix 
* ROC curve 
* AUC value

### Kaggle score
* Kaggle score (Screen Shot)

## 5. Discussion and conclusion
* Why did the new methods work better?
* Actually, the simple algorithm KNN with some specific distance made a result as well as PCA!
* Future work: Text mining, Compressed sensing, Factorization Machines (2010), Latent Dirichlet Allocation.

<!--
* KNN: d(id1,id2) = #{ing(id1) \neq ing(id2)}
	* Preprocessing -> none
	* Dimension reduction -> none
	* Training
	* Tuning -> How to choose K? 
	* Testing -> Best K=21, S=0.67659
-->

## Appendix
### Coding
* You can find all the codes in this site. [https://github.com/alicia6174/Kaggle-Whats-Cooking] 
* Please contact me if you have any question.
<yhguan8128@gmail.com>

[https://github.com/alicia6174/Kaggle-Whats-Cooking]: https://github.com/alicia6174/Kaggle-Whats-Cooking 

### How to use Weka & Version of Weka
Sketch the steps.

* Transform training data to a csv file. * Try several multi-class classifications and choose features (ex. #(ing.) for each cuisine).* Compute the accuracy and cross validation.* Choose a model to test. ex. J48, KNN, PLA, Bayes...

Formal steps.* Create a new csv data file (in a needed form).* Convert it to UTF8 encoding. (use instruction in vim [A]).* Convert it into train and test arff files (Hsin’s shell-script [A]).* Train train.arff  by xxx on Weka, analyze the data (MAE, ROC…), and save xxx.model.* Test test.arff by the model, and save result_xxx.txt.* Convert result_xxx.txt to result_xxx.csv.

### How to use Automatic Weka & & Version of AutoWeka* For the newest Weka 3.8.0 and Auto-Weka 2.5, it needs to install Java and JDK and type the specific instruction to avoid java executable issue. 
## References
<!--
1. P. Hall *et al.* Choice of neighbor order in nearest-neighbor classification. *Ann. Stat.*, 36(5):2135-2152, 2008.
[1]: https://arxiv.org/pdf/0810.5276.pdf 
-->

2. Y.-J. Lee, Y.-R. Yeh, and H.-K. Pao. Introduction to Support Vector Machines and Their Applications in Bankruptcy Prognosis. *Handbook of Computational Finance*, 731-761, 2012.
[2]: https://link.springer.com/chapter/10.1007%2F978-3-642-17254-0_27

3. 袁梅宇. *王者歸來：WEKA機器學習與大數據聖經 3/e.* 佳魁資訊, 2016.
[3]: https://www.tenlong.com.tw/products/9789863794578 

<!-- §1. 寫完§2,3,4後Introduction需要修正 -->
<!-- §2,3. 重新命名檔案！加上每個演算法的選取參數！有底線的地方要打成"\_"! -->
<!-- §5. KNN與其他方法資料型態不一樣且沒有evaluation的部分 -> 了解各方法的細節看如何寫比較好且補足需要的部分或者乾脆省略 -->
<!-- 新的section新起一頁？ -->
<!-- 參考資料recheck, 目錄, 頁碼, 與插圖？ -->
