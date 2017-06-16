&nbsp; <p>
&nbsp; <p>
# <center> Machine Learning Kaggle Project </center>
# <center> What's Cooking </center>
&nbsp; <p>
&nbsp; <p>
&nbsp; <p>
<center> <img src="./pictures/Whats_cooking.png" width="50%" /> </center>
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
| **Total number of ingregients in training data** | **6714** |

</center>

<!--
| **Length of longest list of ingregients in training data** | **65** |
| **Length of shortest list of ingregients in training data** | **1** |
-->

The submission needs to be saved as a csv file of this form.
<center>

| id | cuisine |
| :---: | :---: |
| 18009 | italian |
| 35203 | chinese |
| ... | ... |

</center>

Obviously, this competition is a **supervised problem** of **multi-class classification**. There are several kinds of applicable classifiers.

* Transform to binary - OvR (one-against-all), OvO (one-against-one)* Extend from binary - Naive Bayes, KNN (IBk), Decision trees (J48), SVM (SMO), Neural networks (Multilayer Perceptron)
### The main steps of learning
There are five major steps of machine learning in this project.

* **Data analysis** - We started with **data preprocessing** by these initial process.
	* Delete these special characters: ç, è, é, ®, and ™.
	* Convert all the strings into the type of UTF-8. 

* **Visualization** - If we transform the training data into a sparse matrix full of $0$ and $1$ directly, the matrix will have the size of $39774 \times 6714$. To cope with this matrix more efficiently, we need **dimension reduction** to compress the size of matrix without losing too many varieties of data. **This step is the main difference between related work and our new method** which will be mentioned in the sections 2 \& 3.

* **Modeling** - We chose **Weka** environment to create models. The detailed process of converting data matrix to the valid file for Weka environment will be mentioned also in the sections 2 \& 3. All the codes used in this project can be found in the GitHub.
	* [https://github.com/alicia6174/Kaggle-Whats-Cooking]

	The instructions of how to use Weka will be described in the appendix.
	We skipped **tuning** in this project because we focused on comparing the results of two methods under different models.

<!--
**AutoWeka** is a tool which provides the ideal model for any given data, but we didn't use it in this project since it cost too much time. 
-->
	
<!--
In the section 3, we defined a specific score to determine the number of principal components used to reduce the dimension.
Can we use AutoWeka to do tuning? 
-->
* **Evaluation** - We presented various quantitiest to evaluate different models in the section 4.  Finally, we gave the scores of our submissions on Kaggle site.

<!--
These evidences showed that our method works better than the old one.
-->

* **Prediction** - We saved the best model depending on the evaluation and used it to predict the cuisine of each testing data. The process will be mentioned also in the sections 2 \& 3.

<!--
### Data preprocessing 
* delete special characters(appendix) 
* convert to unitcode(appendix) 
* change to 01-vectors 
* a matrix of size 39774 x 6714 (training) without missing values -> very large!

Detailed steps.

* prefixFilter			create train.json  → delete special characters
* :set ff=unix			unicode
* a.encode(‘utf-8’)		unicode
-->

## 2. Related work
### Descriptions of method

* **Dimension reduction** - The old method collected the **top ingredients** which occur most frequently in the training data as the features. To compare with our method, we chose the number of features to be $1000$. In that way, each data could be transformed into a $1000$-dimensional vector with the $i$th component being $1$ if its ingredients contain the $i$th feature and being $0$ if otherwise. The training data matrix of size $39774 \times 1000$ (without the header and labels) had this form and was saved as a csv file.

<center>

| 1 | 2 | $\ldots\ldots$ | 1000 | cuisine |
| :---: | :---: | :---: | :---: | :---: |
| 0 | 0 | $\ldots\ldots$ | 0 | greek |   
| 1 | 0 | $\ldots\ldots$ | 0 | southern_us |
| $\vdots$ | $\vdots$ | $\ldots\ldots$ | $\vdots$ | $\vdots$ |
| 0 | 1 | $\ldots\ldots$ | 1 | mexican |

</center>
 
* **Modeling** - We converted the csv file into an arff file so that the Weka environment would work more smoothly. We tried several multi-class classifiers for comparison. According to the evaluation (see §4), we saved the best model   **SMO** to make predictions.

* **Prediction** - We repeated the steps of preprocessing and converting files to create the arff file of testing data. Then we used Weka again to predict the result. Finally, we saved the result as a needed submission file and uploaded it on the Kaggle site for scoring.

* Matrix
### Detailed steps<center>
| Coding files \& ML Tools | Created files | Goals |
| :--- | :--- | :--- |
| prefixFilter | train.json | delete special characters |
| create\_top\_ing.py | ing\_top1000.csv | find top 1000 ingredietns |
| create_weka.py | train\_weka\_top1000.csv **(81M)** | create the reduced training data for modeling |
| weka-csv-arff.pl | train\_weka\_top1000.arff | convert to arff file | 
| Weka |  | create models and make evaluations |
| Weka | train\_weka\_top1000\_SMO.model | create the model of SMO |
| prefixFilter | test.json | delete special characters |
| create\_weka.py | test\_weka\_top1000.csv **(???M)** | create the reduced testing data for prediction |
| weka-csv-arff.pl | test\_weka\_top1000.arff | convert to arff file |
| Weka | test\_weka\_top1000\_SMO.txt | make predictions |
| weka-to-kaggle.pl | test\_weka\_top1000\_SMO.csv | create the submission file for Kaggle |

</center>

The 1001th attribute in the file test\_weka\_top1000.arff needs to be modified to the $20$ cuisines before testing.

## 3. New methods
### Descriptions of method

* **Dimension reduction** - Our method adopted **PCA** which is a linear unsupervised reduction. First we collected the totally $6714$ ingredients as features and each data could be transformed into a $6714$-dimensional vector with the $i$th component being $1$ if its ingredients contain the $i$th feature and being $0$ if otherwise. In that way, we could create the training data matrix of size $39774 \times 6714$. Second we computed the eigenvalues and eigenvectors of the corresponding covariance matrix. Third we chose the number of reduced dimension to be $1000$ according to the score defined by
$$\textrm{Score}(k) = \frac{\sum_{i=1}^k \lambda_i}{\sum_{i=1}^{6714} \lambda_i}$$
where $\lambda_i$s are the eigenvalues which satisfy $\lambda_1 \geq \lambda_2 \geq \ldots \geq \lambda_{6714}$. This grapf of score versus number of eigenvalues shows that $1000$ corresponds to the score of $90$.
<center> <img src="./pictures/Score.pdf" width="80%" /> </center>
Finally we multiply the training data matrix by this matrix composed of the top $1000$ eigenvectors to obtain the reduced training data matrix.
$$
\begin{bmatrix}
\quad & \quad & \quad & \quad \\
\quad & \quad & \quad & \quad \\
v_1 & v_2 & \ldots & v_{1000}\\
\quad & \quad & \quad & \quad \\
\quad & \quad & \quad & \quad 
\end{bmatrix}_{\; 6714 \times 1000}
$$
The training data matrix of size $39774 \times 1000$ (without the header and labels) had this form and was saved as a csv file.

<center>

| 1 | 2 | $\ldots\ldots$ | 1000 | cuisine |
| :---: | :---: | :---: | :---: | :---: |
| 0 | 0 | $\ldots\ldots$ | 0 | greek |   
| 1 | 0 | $\ldots\ldots$ | 0 | southern_us |
| $\vdots$ | $\vdots$ | $\ldots\ldots$ | $\vdots$ | $\vdots$ |
| 0 | 1 | $\ldots\ldots$ | 1 | mexican |

</center>

* PCA 1000 PCs (normalized and rounded),  Matrix.

* **Modeling** - This step was conducted almost the same as in the section 2. The main difference was that **SMO** still served as the best model after evaluation (see §4).

* **Prediction** - 

* Matrix

### Detailed steps
<center>

| Coding files \& ML Tools | Created files | Goals |
| :--- | :--- | :--- |
| prefixFilter | train.json | delete special characters |
| create_mtx.py | train_mtx.csv | create the training data matrix of size 39774 x 6714 |
| do\_pca.cpp | eigVal_eiglVec | find the PCs and eigenvalues of the above matrix |
| create\_eigVec.pl | eigVec | divide the file eigVal_eiglVec into eigVec and eigVal |
| create\_eigVal.pl | eigVal | divide the file eigVal_eiglVec into eigVec and eigVal |
| create\_pca_mtx.m | train\_pca_mtx\_1000.csv | create the reduced training data matrix of size 39774 x 1000 by matrix mutiplication |
| create\_weka.py | train\_weka\_pca1000.csv **(187M)** | create the reduced training data for modeling |
| weka-csv-arff.pl | train\_weka\_pca1000.arff | convert to arff file |
| Weka |  | create models and make evaluations |
| Weka | train\_weka\_pca1000\_SMO.model | create the model of SMO |
| prefixFilter | test.json | delete special characters |
| create\_mtx.py | test\_mtx.csv | create the testing data matrix of size 9944 x 6714 |
| create\_pca\_mtx.m | test\_pca\_mtx\_1000.csv | create the reduced testing data matrix of size 9944 x 1000 by matrix mutiplication |
| create\_weka.py | test\_weka\_pca1000.csv **(48.7M)** | create the reduced testing data for prediction |
| weka-csv-arff.pl | test\_weka\_pca1000.arff | convert to arff file |
| Weka | test\_weka\_pca1000\_SMO.txt | make predictions |
| weka-to-kaggle.pl | test\_weka\_pca1000\_SMO.csv | create the submission file for Kaggle |

</center>

The 1001th attribute in the file test\_weka\_pca1000.arff needs to be modified to the 20 cuisines before testing.

## 4. Comparison results
### Evaluation
* Put some of these results in the appendix since the matrix of results are too large!
* The device we used is Google * 4. We run four PC simultaneously for convenience.
* Correctness / Accuracy / Error rate. We computed Percent 66% instead of K-fold cross-validation since the data is too large!

	Old method:
	
	* Naïve Bayes					63.3365 %
	* IBk, k=1501					31.7533 % Take k=1501 for some reason
	* SMO							xxx %
	* MultilayerPerceptron		??? %
	* J48							64.2387 %
	* MultiClassClassifier(OvR)??? %
	* MultiClassClassifier(OvO)???

	New method:
	
	* Naïve Bayes					3x.xxx %
	* IBk, k=1501					30.8364 % Take k=1501 for some reason
	* SMO							73.2382 %
	* MultilayerPerceptron		??? %
	* J48							40.0281 %
	* MultiClassClassifier(OvR)??? % (pink)
	* MultiClassClassifier(OvO)Out of memory!
* Training time
* Confusion matrix 
* ROC, AUC ...

### Kaggle score
* Old: Top 1000, S=???
* New: PCs 1000, S=0.66020
* Screen Shot of Kaggle score

## 5. Discussion and conclusion
* The file size 81M vs. 187M due to the float type of PCA data.
* Why did we choose the number of features to be 1000? Score.pdf
	* Old method: We've tried Top 200 ing. + ing_len (normalized) -> Not good enough.
	* New method: We've tried PCA 2000 (normalized) -> Out of memory. 
* Why did we choose 66 % instead of k-fold validation? The data is too large.
* Why did we choose those models?
* The parameter K of KNN was taken as K=1501 due to the reference. Actually, we made a KNN algorithm ourselves with some specific distance and K=21 created a better result! (Screen Shot of Kaggle score)
You can find the code also in our GitHub site.
	* [https://github.com/alicia6174/Kaggle-Whats-Cooking]
* The reslut of the new method seemed the same as the old one. We don't know why.. 
* Future work - Text mining, Compressed sensing, Factorization Machines (2010), Latent Dirichlet Allocation.

<!--
* KNN: d(id1,id2) = #{ing(id1) \neq ing(id2)}
	* Preprocessing -> none
	* Dimension reduction -> none
	* Training
	* Tuning -> How to choose K? 
	* Testing -> Best K=21, S=0.67659
-->

## Appendix
<!--
### Coding
* You can find all the codes in this site. [https://github.com/alicia6174/Kaggle-Whats-Cooking] 
* Please contact me if you have any question.
<yhguan8128@gmail.com>

[https://github.com/alicia6174/Kaggle-Whats-Cooking]: https://github.com/alicia6174/Kaggle-Whats-Cooking 
-->

### How to use Weka
Version of Weka.

* 3.8.0
* Need to install Java

Sketch the steps.

* Transform training data to a csv file. * Try several multi-class classifications and choose features (ex. #(ing.) for each cuisine).* Compute the accuracy and cross validation.* Choose a model to test. ex. J48, KNN, PLA, Bayes...

Formal steps.* Create a new csv data file (in a needed form).* Convert it to UTF8 encoding. (use instruction in vim [A]).* Convert it into train and test arff files (Hsin’s shell-script [A]).* Train train.arff  by xxx on Weka, analyze the data (MAE, ROC…), and save xxx.model.* Test test.arff by the model, and save result_xxx.txt.* Convert result_xxx.txt to result_xxx.csv.

### How to use Automatic Weka & & Version of AutoWekaVersion of Weka.* For the newest Weka 3.8.0 and Auto-Weka 2.5, it needs to install Java and JDK and type the specific instruction to avoid java executable issue. Sketch the steps.

* ...
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
<!-- §2,3. 重新命名檔案！加上每個演算法的選取參數！有底線的地方要打成"\_"! create\_eigVec.pl與create\_eigVal.pl檢查是否正確！ -->
<!-- §5. KNN與其他方法資料型態不一樣且沒有evaluation的部分 -> 了解各方法的細節看如何寫比較好且補足需要的部分或者乾脆省略 -->
<!-- 表格變色了...-->
<!-- 新的section新起一頁？ -->
<!-- 參考資料recheck, 目錄, 頁碼, 與插圖？ -->
<!-- 重要的:為何只看ROC, AUC...?  -->
