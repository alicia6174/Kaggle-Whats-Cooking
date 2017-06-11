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

* **Data analysis** - We dealt with **data preprocessing** and decided the form of data representation in this step, which will be mentioned later in this subsection.
* **Visualization** - According to the above result, the data matrix derived from the training instances has size of 39774 x 6714. To cope with this matrix more efficiently, we need **dimension reduction** to compress the size of data without losing too many varieties of data. The methods mentioned in the sections 2 and 3 adopted different ways to achieve dimension reduction.
* **Modeling** - We chose **Weka** environment to create models. The process of converting data matrix to the valid file for Weka environment will be mentioned in the sections 2 and 3. The detailed steps of how to use Weka will be described in the appendix.
* **Tuning** - In the section 3, we defined a special score to determine the number of principal components used to reduce the dimension. And then, we used the  **AutoWeka** tool to choose appropriate parameters needed in the candidate models. The detailed steps of how to use AutoWeka will be described in the appendix.
* **Evaluation** - In the section 4, we presented various quantitiest to evaluate old method and our method. These evidences showed that our method works better than the old one. Finally, we gave the scores of our submissions on Kaggle site.

### Data preprocessing 
* delete special characters(appendix) 
* convert to unitcode(appendix) 
* change to 01-vectors 
* a matrix of size 39774 x 6714 (training) without missing values -> very large!

## 2. Related work
* Top ing. + J48 -> sketch the method
* Dimension reduction -> Top ing.
* Modeling -> Weka (detailed steps)
* Tuning -> How to choose num. of top ing.?
* Best: Top 1000, S=0.57; Top 1703, S=??? 

## 3. New methods
* PCA + SMO -> sketch the method
* Dimension reduction -> PCA, linear unsupervised reduction
* Modeling -> Auto Weka + Weka (detailed steps)
* Tuning -> 1. How to choose num. of PCs? Score.pdf 2. How to choose parameters of SMo? AutoWeka
* Best: PCs 1000, S=0.66020; PCs 1703, S=???

## 4. Comparison results
### Evaluation
* Correctness / Accuracy / Error rate
* K-fold cross-validation
* Confusion matrix
* ROC curve 
* AUC value

### Testing
* Testing -> Weka (detailed steps)
* Kaggle score (Screen Shot)

## 5. Discussion and conclusion
* Why did the new methods work better?
* Future work: Text mining, Compressed sensing, Factorization Machines (2010), Latent Dirichlet Allocation.

## Appendix
<!--
* KNN: d(id1,id2) = #{ing(id1) \neq ing(id2)}
	* Preprocessing -> none
	* Dimension reduction -> none
	* Training
	* Tuning -> How to choose K? 
	* Testing -> Best K=21, S=0.67659
-->
* How to use Weka & Version of Weka
* How to use Automatic Weka & & Version of AutoWeka* Coding
	* You can find all the codes in this site. [https://github.com/alicia6174/Kaggle-Whats-Cooking] 
	* Please contact this mail if you have any question.
<yhguan8128@gmail.com>

[https://github.com/alicia6174/Kaggle-Whats-Cooking]: https://github.com/alicia6174/Kaggle-Whats-Cooking 

## References
<!--
1. P. Hall *et al.* Choice of neighbor order in nearest-neighbor classification. *Ann. Stat.*, 36(5):2135-2152, 2008.
[1]: https://arxiv.org/pdf/0810.5276.pdf 
-->

2. Y.-J. Lee, Y.-R. Yeh, and H.-K. Pao. Introduction to Support Vector Machines and Their Applications in Bankruptcy Prognosis. *Handbook of Computational Finance*, 731-761, 2012.
[2]: https://link.springer.com/chapter/10.1007%2F978-3-642-17254-0_27

3. 袁梅宇. *王者歸來：WEKA機器學習與大數據聖經 3/e.* 佳魁資訊, 2016.
[3]: https://www.tenlong.com.tw/products/9789863794578 

<!-- 表格顏色怪怪 -> 考慮用html調整表格與上一行文字之間留白部分 // 轉成pdf就不會灰白變色了 -->
<!-- 新的section新起一頁？ -->
<!-- 參考資料recheck, 目錄, 頁碼, 與插圖？ -->
<!-- KNN與其他方法資料型態不一樣且沒有evaluation的部分 -> 了解各方法的細節看如何寫比較好且補足需要的部分或者乾脆省略 -->
