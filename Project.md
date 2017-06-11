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
### <center> yhguan8128@gmail.com </center>

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

* Transform to binary: OvR, OvO* Extend from binary: Naive Bayes, KNN, Decision trees, SVM, Neural networks, ELM* Hierarchical classification

### The main steps of learning
* Preprocessing(§1) - Training(§2,3) - Evaluating(§4) - Testing(§4)
* Preprocessing: Data representation in this project

## 2. Related work
### Top ing. + J48
* Detailed steps of training

## 3. New methods
### KNN
* Detailed steps of training

### PCA + SMO
* Detailed steps of training

## 4. Comparison results
### Evaluating
* Correctness
* K-fold cross-validation
* Confusion matrix
* ROC curve 
* AUC value

### Testing
* Detailed steps of testing
* Score (illustration)

## 5. Discussion and conclusion
* Why did the new methods work better?

## Appendix
### How to use Weka
### Automatic Weka### All the codes in this project
https://github.com/alicia6174/Kaggle-Whats-Cooking 
[code]: https://github.com/alicia6174/Kaggle-Whats-Cooking 

## References
1. P. Hall *et al.* Choice of neighbor order in nearest-neighbor classification. *Ann. Stat.*, 36(5):2135-2152, 2008.
[1]: https://arxiv.org/pdf/0810.5276.pdf 

2. Y.-J. Lee, Y.-R. Yeh, and H.-K. Pao. Introduction to Support Vector Machines and Their Applications in Bankruptcy Prognosis. *Handbook of Computational Finance*, 731-761, 2012.
[2]: https://link.springer.com/chapter/10.1007%2F978-3-642-17254-0_27

3. 袁梅宇. *王者歸來：WEKA機器學習與大數據聖經 3/e.* 佳魁資訊, 2016.
[3]: https://www.tenlong.com.tw/products/9789863794578 

<!-- 表格顏色怪怪 -->
<!-- 確認我跟教授論文的結構是否一樣 Preprocessing(§1) - Train(§2,3) - Evaluating & Testing (§4) -->
<!-- subsection要編號？ -->
<!-- 新的section新起一頁？ -->