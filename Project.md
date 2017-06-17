&nbsp; <p>
&nbsp; <p>
# <center> Final Project on Kaggle Competition </center>
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
### <center> June 21, 2017 </center>

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
<table border="1" align="center">
<tr align="center"> <td>irish</td> <td>mexican</td> <td>chinese</td> <td>filipino</td> <td>vietnamese</td> </tr>
<tr align="center"> <td>moroccan</td> <td>brazilian</td> <td>japanese</td> <td>british</td> <td>greek</td> </tr>
<tr align="center"> <td>indian</td> <td>jamaican</td> <td>french</td> <td>spanish</td> <td>russian</td> </tr>
<tr align="center"> <td>cajun\_creole</td> <td>thai</td> <td>southern_us</td> <td>korean</td> <td>italian</td> </tr>
</table>
</center>

After an initial step of statistics, some basic summaries about data are given below.

<center>
<table border="1" align="center">
<tr align="center"> <td>**Number of training instances**</td> <td>39774</td> </tr>
<tr align="center"> <td>**Number of testing instances**</td> <td>9944</td> </tr>
<tr align="center"> <td>**Total number of ingregients in training data**</td> <td>6714</td> </tr>
</table>
</center>

<!--
| **Length of longest list of ingregients in training data** | **65** |
| **Length of shortest list of ingregients in training data** | **1** |
-->

The submission needs to be saved as a csv file of this form.

<center>
<table border="1" align="center">
<tr align="center"> <td>**id**</td> <td>**cuisine**</td> </tr>
<tr align="center"> <td>18009</td> <td>italian</td> </tr>
<tr align="center"> <td>35203</td> <td>chinese</td> </tr>
<tr align="center"> <td>$\vdots$</td> <td>$\vdots$</td> </tr>
</table>
</center>

Obviously, this competition is a **supervised problem** of **multi-class classification**. There are several kinds of applicable classifiers.

* Transform to binary - **OvR** (one-against-all), **OvO** (one-against-one)* Extend from binary - **Naïve Bayes**, **KNN** (IBk), **Decision trees** (J48), **SVM** (SMO), **Neural networks** (Multilayer Perceptron)
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
Can we use AutoWeka to do tuning? 
-->

* **Evaluation** - We presented various quantitiest to evaluate different models in the section 4.  Finally, we gave the scores of our submissions on Kaggle site.

<!--
These evidences showed that our method works better than the old one.
-->

* **Prediction** - We saved the best model depending on the evaluation and used it to predict the cuisine of each testing data. The process will be mentioned also in the sections 2 \& 3.

## 2. Related work
### Descriptions of method

* **Dimension reduction** - The old method collected the **top ingredients** which occur most frequently in the training data as the features. To compare with our method, we chose the number of features to be $1000$. In that way, each data could be transformed into a $1000$-dimensional vector with the $i$th component being $1$ if its ingredients contain the $i$th feature and being $0$ if otherwise. The training data matrix of size $39774 \times 1000$ (without the header and labels) had this form and was saved as a csv file.

<center>
<table border="1" align="center">
<tr align="center"> <td>**1**</td> <td>**2**</td> <td>$\ldots\ldots$</td> <td>**1000**</td> <td>**cuisine**</td> </tr>
<tr align="center"> <td>0</td> <td>0</td> <td>$\ldots\ldots$</td> <td>0</td> <td>greek</td> </tr>
<tr align="center"> <td>1</td> <td>0</td> <td>$\ldots\ldots$</td> <td>0</td> <td>southern_us</td> </tr>
<tr align="center"> <td>$\vdots$</td> <td>$\vdots$</td> <td>$\ldots\ldots$</td> <td>$\vdots$</td> <td>$\vdots$</td> </tr>
<tr align="center"> <td>0</td> <td>1</td> <td>$\ldots\ldots$</td> <td>1</td> <td>mexican</td> </tr>
</table>
</center>
 
* **Modeling** - We converted the csv file into an arff file so that the Weka environment would work more smoothly. We tried several multi-class classifiers for comparison. According to the evaluation (see §4), we saved the best model   **SMO** to make predictions.

* **Prediction** - We repeated the steps of preprocessing and file conversion to create the arff file of testing data. Then we used Weka again to predict the result. Finally, we saved the result as a needed submission file and uploaded it on the Kaggle site for scoring.

* Matrix???
### Detailed steps
<center>
<table border="1" align="center">
<tr align="left"> <td>**Codes \& ML Tool**</td> <td>**Created files**</td> <td>**Goals**</td> </tr>
<tr align="left"> <td>prefix\_filter</td> <td>train.json</td> <td>delete special characters</td> </tr>
<tr align="left"> <td>create\_top\_ing.py</td> <td>ing\_top1000.csv</td> <td>find top 1000 ingredietns</td> </tr>
<tr align="left"> <td>create_weka.py</td> <td>train\_weka\_top1000.csv **(81M)**</td> <td>create the reduced training data for modeling</td> </tr>
<tr align="left"> <td>weka-csv-arff.pl</td> <td>train\_weka\_top1000.arff</td> <td>convert to arff file</td> </tr>
<tr align="left"> <td>Weka</td> <td></td> <td>create models and make evaluations</td> </tr>
<tr align="left"> <td>Weka</td> <td>train\_weka\_top1000\_SMO.model</td> <td>create the model of SMO</td> </tr>
<tr align="left"> <td>prefix\_filter</td> <td>test.json</td> <td>delete special characters</td> </tr>
<tr align="left"> <td>create\_weka.py</td> <td>test\_weka\_top1000.csv **(21.2M)**</td> <td>create the reduced testing data for prediction</td> </tr>
<tr align="left"> <td>weka-csv-arff.pl</td> <td>test\_weka\_top1000.arff</td> <td>convert to arff file</td> </tr>
<tr align="left"> <td>Weka</td> <td>test\_weka\_top1000\_SMO.txt</td> <td>make predictions</td> </tr>
<tr align="left"> <td>weka-to-kaggle.pl</td> <td>test\_weka\_top1000\_SMO.csv</td> <td>create the submission file for Kaggle</td> </tr>
</table>
</center>

The 1001th attribute in the file test\_weka\_top1000.arff needs to be modified to the $20$ cuisines before testing.

## 3. New methods
### Descriptions of method

* **Dimension reduction** - Our method adopted **PCA** which is a linear unsupervised reduction. First we collected the totally $6714$ ingredients as features and each data could be transformed into a $6714$-dimensional vector with the $i$th component being $1$ if its ingredients contain the $i$th feature and being $0$ if otherwise. In that way, we could create the training data matrix of size $39774 \times 6714$. Second we computed the eigenvalues and eigenvectors of the corresponding covariance matrix. Third we chose the number of reduced dimension to be $1000$ according to the score defined by
$$\textrm{Score}(k) = \frac{\sum_{i=1}^k \lambda_i}{\sum_{i=1}^{6714} \lambda_i}$$
where $\lambda_i$s are the eigenvalues which satisfy $\lambda_1 \geq \lambda_2 \geq \ldots \geq \lambda_{6714}$. This grapf of score versus number of eigenvalues shows that $1000$ corresponds to the score of $90$.
<center> <img src="./pictures/Score.pdf" width="80%" /> </center>
Finally we multiplied the training data matrix by this matrix composed of the top $1000$ eigenvectors to obtain the reduced training data matrix. Each feature had been normalized and rounded to the second decimal.
$$
\begin{bmatrix}
\quad & \quad & \quad & \quad \\
\quad & \quad & \quad & \quad \\
v_1 & v_2 & \ldots & v_{1000}\\
\quad & \quad & \quad & \quad \\
\quad & \quad & \quad & \quad 
\end{bmatrix}_{\; 6714 \times 1000}
$$
The reduced training data matrix of size $39774 \times 1000$ (without the header and labels) had this form and was saved as a csv file.

<center>
<table border="1" align="center">
<tr align="center"> <td>**1**</td> <td>**2**</td> <td>$\ldots\ldots$</td> <td>**1000**</td> <td>**cuisine**</td> </tr>
<tr align="center"> <td>0.71</td> <td>0.34</td> <td>$\ldots\ldots$</td> <td>0.45</td> <td>greek</td> </tr>
<tr align="center"> <td>0.49</td> <td>0.57</td> <td>$\ldots\ldots$</td> <td>0.47</td> <td>southern_us</td> </tr>
<tr align="center"> <td>$\vdots$</td> <td>$\vdots$</td> <td>$\ldots\ldots$</td> <td>$\vdots$</td> <td>$\vdots$</td> </tr>
<tr align="center"> <td>0.30</td> <td>0.30</td> <td>$\ldots\ldots$</td> <td>0.47</td> <td>mexican</td> </tr>
</table>
</center>

* **Modeling** - This step was conducted almost in the same way as in the section 2. The main difference was that **SMO** still served as the best model after evaluation (see §4).

* **Prediction** - We conducted the same step of dimension reduction to obtain the reduced testing data matrix. The reduced testing data matrix of size $9944 \times 1000$ (without the header and labels) had this form and was saved as a csv file. The following steps was conducted in the same way as in the section 2.

<center>
<table border="1" align="center">
<tr align="center"> <td>**1**</td> <td>**2**</td> <td>$\ldots\ldots$</td> <td>**1000**</td> <td>**cuisine**</td> </tr>
<tr align="center"> <td>0.83</td> <td>0.67</td> <td>$\ldots\ldots$</td> <td>0.52</td> <td>?</td> </tr>
<tr align="center"> <td>0.93</td> <td>0.63</td> <td>$\ldots\ldots$</td> <td>0.53</td> <td>?</td> </tr>
<tr align="center"> <td>$\vdots$</td> <td>$\vdots$</td> <td>$\ldots\ldots$</td> <td>$\vdots$</td> <td>$\vdots$</td> </tr>
<tr align="center"> <td>0.70</td> <td>0.20</td> <td>$\ldots\ldots$</td> <td>0.47</td> <td>?</td> </tr>
</table>
</center>

### Detailed steps
<center>
<table border="1" align="center">
<tr align="left"> <td>**Codes \& ML Tool**</td> <td>**Created files**</td> <td>**Goals**</td> </tr>
<tr align="left"> <td>prefix\_filter</td> <td>train.json</td> <td>delete special characters</td> </tr>
<tr align="left"> <td>create\_top\_ing.py</td> <td>ing.csv</td> <td>find all the 6714 ingredients</td> </tr>
<tr align="left"> <td>create\_mtx.py</td> <td>train\_mtx.csv</td> <td>create the training data matrix of size 39774 x 6714</td> </tr>
<tr align="left"> <td>do\_pca.cpp</td> <td>eigVal\_eiglVec</td> <td>find the PCs and eigenvalues of the above matrix</td> </tr>
<tr align="left"> <td>create\_eigVec.pl</td> <td>eigVec</td> <td>divide the file eigVal\_eiglVec into eigVec and eigVal</td> </tr>
<tr align="left"> <td>create\_eigVal.pl</td> <td>eigVal</td> <td>divide the file eigVal\_eiglVec into eigVec and eigVal</td> </tr>
<tr align="left"> <td>create\_pca_mtx.m</td> <td>train\_pca\_mtx\_1000.csv</td> <td>create the reduced training data matrix of size 39774 x 1000 by matrix mutiplication</td> </tr>
<tr align="left"> <td>create\_weka.py</td> <td>train\_weka\_pca1000.csv **(187M)**</td> <td>create the reduced training data for modeling </td> </tr>
<tr align="left"> <td>weka-csv-arff.pl</td> <td>train\_weka\_pca1000.arff</td> <td>convert to arff file</td> </tr>
<tr align="left"> <td>Weka</td> <td></td> <td>create models and make evaluations</td> </tr>
<tr align="left"> <td>Weka</td> <td>train\_weka\_pca1000\_SMO.model</td> <td>create the model of SMO</td> </tr>
<tr align="left"> <td>prefix\_filter</td> <td>test.json</td> <td>delete special characters</td> </tr>
<tr align="left"> <td>create\_mtx.py</td> <td>test\_mtx.csv</td> <td>create the testing data matrix of size 9944 x 6714</td> </tr>
<tr align="left"> <td>create\_pca\_mtx.m</td> <td>test\_pca\_mtx\_1000.csv</td> <td>create the reduced testing data matrix of size 9944 x 1000 by matrix mutiplication</td> </tr>
<tr align="left"> <td>create\_weka.py</td> <td>test\_weka\_pca1000.csv **(48.7M)**</td> <td>create the reduced testing data for prediction</td> </tr>
<tr align="left"> <td>weka-csv-arff.pl</td> <td>test\_weka\_pca1000.arff</td> <td>convert to arff file</td> </tr>
<tr align="left"> <td>Weka</td> <td>test\_weka\_pca1000\_SMO.txt</td> <td>make predictions</td> </tr>
<tr align="left"> <td>weka-to-kaggle.pl</td> <td>test\_weka\_pca1000\_SMO.csv</td> <td>create the submission file for Kaggle</td> </tr>
</table>
</center>

The 1001th attribute in the file test\_weka\_pca1000.arff needed to be modified to the 20 cuisines before testing.

## 4. Comparison results
### Evaluation
* Our devices were the **Virtual Machine Instances** on **Google Cloud Platform**. We applied for 4 virtual machines of this size.
	* ubuntu16-04
	* 1 vCPU
	* 6.5 GB

	And We run these machiens simultaneously for convenience.

* Our ML tool was **Weka Environment** which needed **Java** and **JDK** to avoid java executable issue. 
	* Weka Environment for Knowledge Analysis Version 3.8.0
	* Java version "1.8.0_121"
	* Java(TM) SE Runtime Environment (build 1.8.0_121-b13)
	* Java HotSpot(TM) 64-Bit Server VM (build 25.121-b13, mixed mode)

* We split the training data into **66.0% for training and the remainder for testing**. We didn't compute the k-fold cross-validation since the training data was too large. 
All the results are shown in the table below.

<center>
<table border="1" align="center">
<tr align="center"> <td>**Models**</td> <td>**The Old Method**</td> <td>**The New Method**</td> </tr>
<tr align="center"> <td>Naïve Bayes</td> <td>63.3365 %</td> <td>???</td> </tr>
<tr align="center"> <td>IBk (k=1501) </td> <td>31.7533 %</td> <td>30.8364 %</td> </tr>
<tr align="center"> <td>SMO</td> <td>xxx %</td> <td>73.2382 %</td> </tr>
<tr align="center"> <td>Multilayer Perceptron</td> <td>??? % (green)</td> <td>??? % (blue)</td> </tr>
<tr align="center"> <td>J48</td> <td>64.2387 %</td> <td>40.0281 %</td> </tr>
<tr align="center"> <td>One-against-all</td> <td>??? % (orange)</td> <td>xxx %</td> </tr>
<tr align="center"> <td>One-against-one</td> <td>??? % (pink)</td> <td>Out of memory</td> </tr>
</table>
</center>
Running time?

* Kappa statistic ≈ 1? ROC area ≈ 1? MAE?
	Show only the real results of SMO from Kaggle. 
	Present the quantities we focused and the meaning of each for other models.
* Confusion matrix ? Show only the real results of SMO from Kaggle.

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
* Why did we choose those models? All the parameters are default except extra discriptions.  
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
### How to use Weka
Sketch the steps.

* Transform training data to a csv file. * Try several multi-class classifications and choose features (ex. #(ing.) for each cuisine).* Compute the accuracy and cross validation.* Choose a model to test. ex. J48, KNN, PLA, Bayes...

Formal steps.* Create a new csv data file (in a needed form).* Convert it to UTF8 encoding. (use instruction in vim [A]).* Convert it into train and test arff files (Hsin’s shell-script [A]).* Train train.arff  by xxx on Weka, analyze the data (MAE, ROC…), and save xxx.model.* Test test.arff by the model, and save result_xxx.txt.* Convert result_xxx.txt to result_xxx.csv.

<!--
### How to use Automatic WekaVersion of AutoWeka.Sketch the steps.-->

## References
<!--
1. P. Hall *et al.* Choice of neighbor order in nearest-neighbor classification. *Ann. Stat.*, 36(5):2135-2152, 2008.
[1]: https://arxiv.org/pdf/0810.5276.pdf 
-->

2. Y.-J. Lee, Y.-R. Yeh, and H.-K. Pao. Introduction to Support Vector Machines and Their Applications in Bankruptcy Prognosis. *Handbook of Computational Finance*, 731-761, 2012.
[2]: https://link.springer.com/chapter/10.1007%2F978-3-642-17254-0_27

3. 袁梅宇. *王者歸來：WEKA機器學習與大數據聖經 3/e.* 佳魁資訊, 台北市, 2016.
[3]: https://www.tenlong.com.tw/products/9789863794578 

<!-- §1. 寫完§2,3,4後Introduction需要修正 -->
<!-- §2,3. 加上每個演算法的選取參數！create\_eigVec.pl與create\_eigVal.pl檢查是否正確！ -->
<!-- 新的section新起一頁？ -->
<!-- Ref(PCA Score, KNN's K), 目錄, 頁碼, 與插圖？ -->
<!-- 重要的:為何只看ROC, AUC...?  -->
