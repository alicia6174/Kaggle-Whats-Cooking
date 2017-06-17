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

* **Prediction** - We repeated the steps of preprocessing and file conversion to create the needed file of testing data. The reduced testing data matrix of size $9944 \times 1000$ (without the header and labels) had the following form and was saved as a csv file. After converting it to an arff file, we used Weka again to predict the result. Finally, we saved the result as a needed submission file and uploaded it on the Kaggle site for scoring. 

<center>
<table border="1" align="center">
<tr align="center"> <td>**1**</td> <td>**2**</td> <td>$\ldots\ldots$</td> <td>**1000**</td> <td>**cuisine**</td> </tr>
<tr align="center"> <td>0</td> <td>0</td> <td>$\ldots\ldots$</td> <td>0</td> <td>?</td> </tr>
<tr align="center"> <td>0</td> <td>0</td> <td>$\ldots\ldots$</td> <td>0</td> <td>?</td> </tr>
<tr align="center"> <td>$\vdots$</td> <td>$\vdots$</td> <td>$\ldots\ldots$</td> <td>$\vdots$</td> <td>$\vdots$</td> </tr>
<tr align="center"> <td>0</td> <td>0</td> <td>$\ldots\ldots$</td> <td>0</td> <td>?</td> </tr>
</table>
</center>
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
Our devices were the **Virtual Machine Instances** on **Google Cloud Platform**. We applied for 4 virtual machines of this size. And we run these machiens simultaneously for convenience.

* OS: ubuntu16-04
* HDD: 20G
* CPU: 1 vCP
* RAM: 6.5 GB

We used the ML tool **Weka Environment** with the version shown below. In this project, we needed to modify the heap size to 4G. (The default size is 512M.)
	
* Weka Environment for Knowledge Analysis Version 3.8.0
* Java version "1.8.0_121"
* Java(TM) SE Runtime Environment (build 1.8.0_121-b13)
* Java HotSpot(TM) 64-Bit Server VM (build 25.121-b13, mixed mode)

We split the training data into **66.0% for training and the remainder for testing**. We didn't compute the k-fold cross-validation since the training data was too large. The correctness and the running time are shown in the table below.

<center>
<table border="1" align="center">
<tr align="center"> 
<td colspan="3" valign="center">
**66% Percentage Split Correctness, $\%$** <p> 
**Running Time, $sec.$** 
</td> 
</tr>
<tr align="center"> 
<td>**Models**</td> 
<td>**The Old Method**</td> 
<td>**The New Method**</td> 
</tr>
<tr align="center"> 
<td>IBk (k=1501) </td> 
<td>31.7533 <p> 3093.23</td> 
<td>30.8364 <p> 3451.79</td> 
</tr>
<tr align="center"> 
<td>Naïve Bayes</td> 
<td>63.3365 <p> 92.67</td> 
<td>37.2846 <p> 55.37</td> 
</tr>
<tr align="center"> 
<td>J48</td> 
<td>64.2387 <p> 5337.63</td> 
<td>40.0281 <p> 598.85</td> 
</tr>
<tr align="center"> 
<td>SMO</td> 
<td>73.3417 <p> 3140.75</td> 
<td>73.2382 <p> 2401.43</td> 
</tr>
</table>
</center>

<!--
<tr align="center"> 
<td>One-against-all</td> 
<td>Have run 27.5hr <p> Give up</td> 
<td>66.2797 <p> 12065.99</td> 
</tr>
<tr align="center"> 
<td>One-against-one</td> 
<td>Have run 27hr <p> Give up</td> 
<td>Out of memory</td> 
</tr>
<tr align="center"> 
<td>Multilayer Perceptron</td> 
<td>Have run 27hr <p> Give up</td> 
<td>Have run 30hr <p> Give up</td> 
</tr>
-->

We also focused on the quantities - Kappa statistic, MAE, AUC, and confusion matrix.
	
* **Kappa statistic** $K$ shows the difference between the classifier and stochastic classification, which is a decimal in $[0,1]$. $K=0$ means no difference while $K=1$ represents the classifier is totally different from the stochastic classification. Generally speaking, $K$ is proportional to AUC and correctness. Therefore, the closer $K$ approaches $1$ ($K \approx 1$), the better the result of the classifier is. 
* **Mean absolute error** is the average of absoluate error.
$$ \textrm{MAE} = \frac{\sum_{i=1}^{9944}|e_i|}{9944} $$ 
* **ROC Area** is the area under the ROC curve which is a decimal in $[0,1]$. The closer AUC approaches $1$ ($\textrm{AUC} \approx 1$), the better the result of the classifier is. For multi-class classification problem, we need to evaluate AUC for each class respectively. 
* **Confusion Matrix** is the matrix defined by
$$
(i,j)\textrm{-entry = number of counts for actual class is $i$th class and predicted class is $j$th class.}
$$
Therefore, the more dominated the diagonal is, the better the result of the classifier is.
	
The Kappa statistic and MAE are shown in the following table. The results of AUC  and confusion matrix can be found in the appendix.

<center>
<table border="1" align="center">
<tr align="center"> 
<td colspan="3" valign="center">
**Kappa statistic** <p> 
**Mean absolute error** 
</td> 
</tr>
<tr align="center"> 
<td>**Models**</td> 
<td>**The Old Method**</td> 
<td>**The New Method**</td> 
</tr>
<tr align="center"> 
<td>IBk (k=1501) </td> 
<td>0.1668 <p> 0.0842</td> 
<td>0.1480 <p> 0.0884</td> 
</tr>
<tr align="center"> 
<td>Naïve Bayes</td> 
<td>0.5934 <p> 0.0400</td> 
<td>0.3340 <p> 0.0628</td> 
</tr>
<tr align="center"> 
<td>J48</td> 
<td>0.5998 <p> 0.0425</td> 
<td>0.3351 <p> 0.0607</td> 
</tr>
<tr align="center"> 
<td>SMO</td> 
<td>0.7019 <p> 0.0905</td> 
<td>0.7012 <p> 0.0905</td> 
</tr>
</table>
</center>

### Kaggle score
The detailed process of testing are described in the sections 2 \& 3. The followings are the final results.

* Top-ing Method **0.75030** 
<center> <img src="./pictures/test_weka_top1000_SMO.png" width="80%" /> </center>
* PCA Method **0.66020**
<center> <img src="./pictures/test_weka_pca1000_SMO.png" width="80%" /> </center>

## 5. Discussion and conclusion
* The file size 81M vs. 187M due to the float type of PCA data.
* Why did we choose the number of features to be 1000? Score.pdf
	* Old method: We've tried Top 200 ing. + ing_len (normalized) -> Not good enough.
	* New method: We've tried PCA 2000 (normalized) -> Out of memory. 
* Why did we choose 66 % instead of k-fold validation? The data is too large.
* Why did we choose those models? All the parameters are default except extra discriptions. We've also run OvR, OvO, and Multilayer Perceptron. They all costed over one day, so we didn't wait for the results.
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

* Transform training data to a csv file. * Try several multi-class classifications and choose features (ex. #(ing.) for each cuisine). (Show the graph of feature distribution)* Compute the accuracy and cross validation.* Choose a model to test. ex. J48, KNN, PLA, Bayes...

Formal steps.* Create a new csv data file (in a needed form).* Convert it to UTF8 encoding. (use instruction in vim [A]).* Convert it into train and test arff files (Hsin’s shell-script [A]).* Train train.arff  by xxx on Weka, analyze the data (MAE, ROC…), and save xxx.model.* Test test.arff by the model, and save result_xxx.txt.* Convert result_xxx.txt to result_xxx.csv.

<!--
### How to use Automatic WekaVersion of AutoWeka.Sketch the steps.-->

### Detailed Accuracy By Class
We only put the results of SMO here since SMO served as the bset model.

### Confusion Matrix
We only put the results of SMO here since SMO served as the bset model.

## References
<!--
1. P. Hall *et al.* Choice of neighbor order in nearest-neighbor classification. *Ann. Stat.*, 36(5):2135-2152, 2008.
[1]: https://arxiv.org/pdf/0810.5276.pdf 
-->

2. Y.-J. Lee, Y.-R. Yeh, and H.-K. Pao. Introduction to Support Vector Machines and Their Applications in Bankruptcy Prognosis. *Handbook of Computational Finance*, 731-761, 2012.
[2]: https://link.springer.com/chapter/10.1007%2F978-3-642-17254-0_27

3. 袁梅宇. *王者歸來：WEKA機器學習與大數據聖經 3/e.* 佳魁資訊, 台北市, 2016.
[3]: https://www.tenlong.com.tw/products/9789863794578 

<!-- §1. Recheck. 把五大步驟裡的文章結構拿到外面寫. 加寫兩個方法的動機: Top ing. \& PCA (後面方法均改名為這樣不要叫新舊方法). -->
<!-- §2,3. create\_eigVec.pl與create\_eigVal.pl檢查是否正確！ -->
<!-- 把上傳的檔案放在GitHub. -->
<!-- 新的section新起一頁？ -->
<!-- Ref(PCA Score, KNN's K), 目錄, 頁碼, 與插圖？ -->

<!--
To include more details, we copy the original results of SMO here.

```
Top-ing Method
TP Rate  FP Rate  Precision  Recall   F-Measure  MCC      ROC Area  PRC Area  Class
0.491    0.005    0.570      0.491    0.528      0.524    0.893     0.362     brazilian
0.556    0.002    0.789      0.556    0.652      0.658    0.908     0.538     jamaican
0.730    0.006    0.714      0.730    0.722      0.716    0.956     0.605     moroccan
0.862    0.016    0.814      0.862    0.837      0.824    0.978     0.787     indian
0.451    0.010    0.547      0.451    0.495      0.485    0.859     0.321     spanish
0.750    0.056    0.622      0.750    0.680      0.641    0.926     0.583     southern_us
0.668    0.006    0.769      0.668    0.715      0.709    0.952     0.627     greek
0.367    0.008    0.493      0.367    0.420      0.415    0.917     0.300     british
0.788    0.017    0.768      0.788    0.778      0.762    0.962     0.707     chinese
0.853    0.066    0.757      0.853    0.803      0.753    0.944     0.737     italian
0.253    0.004    0.413      0.253    0.314      0.318    0.882     0.208     russian
0.412    0.004    0.620      0.412    0.495      0.498    0.930     0.377     irish
0.715    0.009    0.787      0.715    0.749      0.739    0.972     0.692     thai
0.571    0.006    0.643      0.571    0.605      0.599    0.924     0.442     filipino
0.759    0.004    0.803      0.759    0.780      0.777    0.982     0.697     korean
0.875    0.023    0.878      0.875    0.877      0.853    0.968     0.845     mexican
0.484    0.006    0.604      0.484    0.538      0.533    0.937     0.401     vietnamese
0.663    0.009    0.758      0.663    0.707      0.698    0.940     0.616     cajun_creole
0.543    0.033    0.542      0.543    0.543      0.510    0.908     0.406     french
0.607    0.007    0.775      0.607    0.681      0.675    0.952     0.584     japanese
0.733    0.029    0.731      0.733    0.729      0.705    0.944     0.647     Weighted Avg.

```
```
PCA Method
TP Rate  FP Rate  Precision  Recall   F-Measure  MCC      ROC Area  PRC Area  Class
0.497    0.005    0.555      0.497    0.524      0.519    0.900     0.380     brazilian
0.633    0.009    0.574      0.633    0.602      0.595    0.917     0.447     filipino
0.658    0.009    0.691      0.658    0.674      0.665    0.948     0.588     greek
0.767    0.004    0.798      0.767    0.782      0.778    0.977     0.709     korean
0.701    0.011    0.723      0.701    0.712      0.700    0.939     0.622     cajun_creole
0.878    0.017    0.811      0.878    0.843      0.831    0.975     0.782     indian
0.429    0.011    0.524      0.429    0.472      0.461    0.854     0.310     spanish
0.407    0.007    0.497      0.407    0.448      0.442    0.919     0.337     irish
0.508    0.002    0.780      0.508    0.615      0.626    0.900     0.508     jamaican
0.512    0.008    0.548      0.512    0.529      0.521    0.940     0.388     vietnamese
0.774    0.016    0.783      0.774    0.779      0.763    0.962     0.711     chinese
0.307    0.005    0.407      0.307    0.350      0.347    0.880     0.214     russian
0.677    0.007    0.806      0.677    0.736      0.728    0.970     0.690     thai
0.331    0.009    0.445      0.331    0.380      0.373    0.905     0.273     british
0.659    0.004    0.771      0.659    0.711      0.707    0.957     0.618     moroccan
0.757    0.050    0.651      0.757    0.700      0.663    0.929     0.596     southern_us
0.852    0.064    0.763      0.852    0.805      0.756    0.942     0.738     italian
0.556    0.031    0.564      0.556    0.560      0.529    0.908     0.432     french
0.623    0.007    0.768      0.623    0.688      0.681    0.951     0.598     japanese
0.865    0.022    0.883      0.865    0.874      0.850    0.969     0.843     mexican
0.732    0.028    0.730      0.732    0.729      0.705    0.943     0.648     Weighted Avg.
```
-->

<!--
We only show the confusion matrix of SMO here since the matrices have the size $20 \times 20$.

* Top-ing Method

```
a    b    c    d    e    f    g    h    i    j    k    l    m    n    o    p    q    r    s    t   <-- classified as
85    2    0    3    6   17    1    3    1   12    0    0    1    4    0   30    1    3    4    0 |    a = brazilian
3  105    0   12    1   29    0    3    1    8    2    0    3    4    2    8    1    2    4    1 |    b = jamaican
0    1  197   25    6    7    6    0    0   15    1    0    1    0    0    7    1    2    1    0 |    c = moroccan
2    5   30  873    3   15    3    4    2   23    2    1    9    2    1   21    1    0    9    7 |    d = indian
7    0    7    5  162   10    3    3    0   82    1    3    3    2    0   34    0   10   26    1 |    e = spanish
11    4    5   10    8 1103    4   14    7  100   12    9    0    5    0   40    0   57   77    5 |    f = southern_us
0    0    3    9    9   11  270    1    1   75    1    2    0    0    0    6    2    1   13    0 |    g = greek
1    3    1    5    6   69    3  103    1   19    3   19    0    1    0    2    0    1   43    1 |    h = british
1    0    1   11    1   23    2    4  722   11    2    1   24   13   27   14   13    4    9   33 |    i = chinese
5    3    5    8   32   83   33   11    5 2248    4    3    2    3    1   37    1   12  135    3 |    j = italian
3    0    5    3    3   29    0   11    2   23   38    5    1    2    0    6    0    0   18    1 |    k = russian
0    2    1    3    5   46    4   22    0   14    6   93    0    1    1    3    0    1   22    2 |    l = irish
3    1    0   25    0    7    1    1   48    2    0    0  409   10    1    6   49    0    2    7 |    m = thai
9    1    0    4    2   18    0    3   27   13    1    2    6  137    0    5    4    3    3    2 |    n = filipino
0    0    0    1    1    2    0    0   25    5    2    0    3    2  192    5    1    0    1   13 |    o = korean
7    3    6   15   25   95    8    2    7   61    0    0    6    5    1 1912    2   12   17    0 |    p = mexican
4    0    5    7    1    9    0    0   27    4    0    0   49   10    3    4  125    0    0   10 |    q = vietnamese
1    1    1    3    6   87    2    3    2   28    2    0    0    2    0   16    1  350   22    1 |    r = cajun_creole
3    0    8    1   19   92   10   20    0  209   11   11    0    2    0   18    1    3  488    2 |    s = french
4    2    1   50    0   20    1    1   62   16    4    1    3    8   10    3    4    1    7  306 |    t = japanese
```
-->

