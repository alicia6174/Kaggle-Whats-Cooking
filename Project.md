# <center> Machine Learning Kaggle Project </center>
# <center> What's Cooking </center>
<center> <img src="./pictures/Fig1.png" width="30%" /> </center>
#### <center> 101021801 Yu-Hsuan Guan  </center>
#### <center> yhguan8128@gmail.com </center>
#### <center> NTHU Department of Mathematics</center>

<div style="page-break-after: always;"></div>

## Introduction
### Main problem
The main issue of this competition is to classify lists of ingredients into correct kinds of cuisines. Each data is originally represented in the format of JSON. 

```python
{
    "id": 10259,
    "cuisine": "greek",
    "ingredients": [ "romaine lettuce", "black olives", "grape tomatoes", "garlic",
    "pepper", "purple onion", "seasoning", "garbanzo beans", "feta cheese crumbles"]
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
| **Number of training instances** | **9944** |
| **Length of longest list of ingregients** | **65** |
| **Length of shortest list of ingregients** | **1** |

</center>

The submission needs to be saved as a csv file of this form.
<center>

| id | cuisine |
| :---: | :---: |
| 35203 | italian |
| 17600 | chinese |
| ... | ... |

</center>

Obviously, this competition is a **supervised problem** of **multi-class classification**. There are several kinds of applicable classifiers.

* Transform to binary: OvR, OvO* Extend from binary: Naive Bayes, KNN, Decision trees, SVM, Neural networks, ELM* Hierarchical classification

### Formal steps of ML
* Preprocessing - Training & Evaluation - Testing
* Structure of this article

## Appendix
### Detailed steps of training and testing
### Environment of Weka### Special codes
### All the codes in this project
https://github.com/alicia6174/Kaggle-Whats-Cooking 
[code]: https://github.com/alicia6174/Kaggle-Whats-Cooking 

## References
1. P. Hall *et al.* Choice of neighbor order in nearest-neighbor classification. *Ann. Stat.*, 36(5):2135-2152, 2008.
[1]: https://arxiv.org/pdf/0810.5276.pdf 

2. Y.-J. Lee, Y.-R. Yeh, and H.-K. Pao. Introduction to Support Vector Machines and Their Applications in Bankruptcy Prognosis. *Handbook of Computational Finance*, 731-761, 2012.
[2]: https://link.springer.com/chapter/10.1007%2F978-3-642-17254-0_27

3. 袁梅宇. *王者歸來：WEKA機器學習與大數據聖經 3/e.* 佳魁資訊, 2016.
[3]: https://www.tenlong.com.tw/products/9789863794578 
