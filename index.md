![alt text](https://www.nvtt.net/wp-content/uploads/2018/10/wine-tasting.jpg)

## Introduction
This study aims to compare the performance of different supervised learning algorithms to predict wine quality based on wine’s physicochemical test results. Wine quality assessment is crucial in the wine industry and is assessed based on physicochemical tests at laboratories and on sensory tests that rely on human experts opinion on wine preferences [1].

## Background and Motivation
In 2018, over 966 million gallons of wine was consumed in the United States [2]. The lack of understanding of physicochemical properties’ effect on the taste of wine and the behavior of human taste makes wine quality categorization non-trivial [3,4].The complexity in the accuracy of prediction and the importance of wine quality assessment in the wine industry motivate the research presented here.  In this presentation, we will apply and compare the performance of supervised learning algorithms to predict wine quality.

The supervised learning models studied were linear regression, logistic regression, decision tree, random forest  and support vector machine (SVM). The ratio of test data to training data was kept constant at 2 to 1 for each model.

Today, we will:

1. Describe the data used

2. Describe the data manipulation techniques used

3. Method Results:

   - Linear, Polynomial, Ridge, & Lasso Regression
   
   - Logistic Regression

   - Decision Tree & Random Forest

   - Support Vector Machine

4. Comparison of different methods' results

5. Conclusions

## Data Description and Initial Data Exploration
This section presents insights on the wine quality dataset which will be used to model quality predictions. 

### Data Distribution in Dataset by Quality Labels and Wine Type
The dataset consists of **6,463** samples. Each sample had **12** features.

The features considered were: type (red or white), fixed acidity, volatile acidity, citric acid, residual sugar, chlorides, free sulfur dioxide, total sulfur dioxide, density, pH, sulphates and alcohol.

*NOTE*: 

The value 1 corresponds to white wine

The value 2 corresponds to red wine
 
![alt text](pic1.JPG)

There are a lot more white wine samples in the dataset than red wine samples, and we don't really have that much data for qualities that are rated 3 or 9.

### How does the quality change with each dimension?

![alt text](rtrpic1.PNG)
![alt text](rtrpic2.PNG)

These two graphs show dimensions that are pretty constant over the quality range, meaning that they likely won't be very important dimensions in our models.

![alt text](rtrpic3.PNG)
![alt text](rtrpic4.PNG)

These two graphs show dimensions that have large variability over each numerical quality **as well as** over wine type.

### Feature Pairplot
![alt text](feature_plots.png)

The average of each dimension varies based off the type of wine that it is. 

None of the dimensions are linearly separable from each other.

## Initial Data Exploration 
### Principal Componenet Analysis
A dataset with 12 feature pushes us to conduct a Principal Componenet Analysis on the data to consider possibility of dimenstion reduction. Principal componenet analysis result for complete dataset reveals that 0.99 of total variance in dataset is explained by first 11 eigenvalues. PCA result indicated not much of dimension reduction can be achieved by using 2 or 3 principal components. Thus, there are not easily identifiable directions in which signiifcant variance of the data points can be captured. 

[0.3176715, 0.21069888, 0.12999856, 0.08094213, 0.06034393, 0.0510322, 0.04471237, 0.04190673, 0.02934314, 0.02139346, 0.00994418]

![alt text](variance_pcs.PNG)

Using principal component analysis, the spread of data is visualized in 2D and 3D space below.

![alt text](pc_2d.png)

<iframe width="700" height="700" frameborder="0" scrolling="no" src="//plot.ly/~abhilashasaroj/108.embed"></iframe>

## Supervised Learning Models for Wine Quality Prediction
This section presents the performance of several supervised learning models developed using given dataset to predict the wine quality data. Traning to test data split ratio of 66% - 33% is used for validation. K-fold cross validation is used to determine mean accuracy score and confusion matrix is used to visualize the classification by the developed classifiers. 

### Quality Prediction Using Linear Regression and Polynomical Regression
First, let's look at the correlation among features and label('quality'):
![alt text](Cor-1.png)
From the correlation map, we can see that the most correlated feature with 'quality' is 'alchol'.
Then we want to plot the relation between each feature and label:
![alt text](LR%20in%20seaborn.png)
We can see that the linear relation between each features and label is not very good prediction, so we want to do a linear combination of all features of dataset to do the prediction. 
#### (1) Linear Regression
We first split our training and test data into 66% and 34%, then we did the Linear Regression Model to fit our function, here's the outcome of our "true_y vs. predicted_y" :
![alt text](LRpredict.png)

##### the calculated MSE = 0.5409578
##### Test Score = 0.299689

#### (2) Polynomial Regression
First, Let's try fit the function when degree n = 3, the "true_y vs. predicted_y" looks like this:
<img width="388" alt="Screen Shot 2019-07-15 at 12 10 17 PM" src="https://user-images.githubusercontent.com/50888610/61231232-bed1d180-a6f9-11e9-8548-8a0b5db8f5cd.png">


##### the calculated MSE = 0.4848
##### Test Score = 0.3385
it is worthnoting that the degree of polynormial function influece the fitting of model, beacaue we don't want to overfit or underfit the model. so we plot the 'MSE vs. polynormial degree n' to see which degree has the lowest MSE value:
<img width="401" alt="Screen Shot 2019-07-15 at 12 13 22 PM" src="https://user-images.githubusercontent.com/50888610/61231341-f771ab00-a6f9-11e9-83f8-cfaa16ff383f.png">
##### Discussion 
By comparing the MSE between linear regression model and polynomial model, we can conclude that for our dataset, polynomial model (when n = 2) is a little bit better than linear regression model and n = other values.

### Quality Prediction Using Ridge Regression and Lasso Regression
#### (3) Ridge Regression
For Ridge Regression, we add a regulation in the function to reduce the magnitude of the coefficients. Our goal is to find the 
![CodeCogsEqn copy](https://user-images.githubusercontent.com/50888610/61229092-84fecc00-a6f5-11e9-9b12-e9e1caf5f0f0.gif) that can optimize the parameters. ![CodeCogsEqn](https://user-images.githubusercontent.com/50888610/61228984-51bc3d00-a6f5-11e9-818e-9dd169da3edc.gif) If we look at the cofficients for different features in RR here:

![alt text](Rcof.png)
As we increase ![CodeCogsEqn copy](https://user-images.githubusercontent.com/50888610/61229092-84fecc00-a6f5-11e9-9b12-e9e1caf5f0f0.gif), we will see that the magnitude of coefficeints decrease.This is because higher the ![CodeCogsEqn copy](https://user-images.githubusercontent.com/50888610/61229092-84fecc00-a6f5-11e9-9b12-e9e1caf5f0f0.gif)  is, bigger is the penalty and therefore, the magnitude of coefficients are reduced.

if we look at the "MSE vs. ![CodeCogsEqn copy](https://user-images.githubusercontent.com/50888610/61229092-84fecc00-a6f5-11e9-9b12-e9e1caf5f0f0.gif)" :
![alt text](RMSE1.png)
 we can see that MSE will decrease first as ![CodeCogsEqn copy](https://user-images.githubusercontent.com/50888610/61229092-84fecc00-a6f5-11e9-9b12-e9e1caf5f0f0.gif) increases and then it ramp to really high value,the lowest MSE was reached when ![CodeCogsEqn copy](https://user-images.githubusercontent.com/50888610/61229092-84fecc00-a6f5-11e9-9b12-e9e1caf5f0f0.gif)is 100, we also plotted the "score vs. ![CodeCogsEqn copy](https://user-images.githubusercontent.com/50888610/61229092-84fecc00-a6f5-11e9-9b12-e9e1caf5f0f0.gif)":
![alt text](Rscore1.png)

we see for this case, score is presenting an opposite trend as MSE, but it also shows the highest score at ![CodeCogsEqn copy](https://user-images.githubusercontent.com/50888610/61229092-84fecc00-a6f5-11e9-9b12-e9e1caf5f0f0.gif) = 100.

##### at ![CodeCogsEqn copy](https://user-images.githubusercontent.com/50888610/61229092-84fecc00-a6f5-11e9-9b12-e9e1caf5f0f0.gif)a = 100, the MSE = 0.3799 and the Test score = 0.34

#### (4) Lasso Regression

For Lasso Regerssion, ![CodeCogsEqn-2](https://user-images.githubusercontent.com/50888610/61230953-0c9a0a00-a6f9-11e9-9cda-8a1c3f04beff.gif), the regulation method is different with Ridge Regression. in RR, the regulation is related to squared coefficient, however, in LR, the regulation is only related to absolute value of coefficient. Therefore, we are expecting that when we increase the value of ![CodeCogsEqn copy](https://user-images.githubusercontent.com/50888610/61229092-84fecc00-a6f5-11e9-9b12-e9e1caf5f0f0.gif), coefficients are approaching towards 0!!!! therefore, Lasso selects some features while reduce the coefficients of others to zero. if we look at the the cofficients for different features in LR here:
![alt text](Lcof.png)

if we look at the "MSE vs. ![CodeCogsEqn copy](https://user-images.githubusercontent.com/50888610/61229092-84fecc00-a6f5-11e9-9b12-e9e1caf5f0f0.gif)" :
![alt text](LMSE.png)
 we can see that MSE keeps increase as we increase ![CodeCogsEqn copy](https://user-images.githubusercontent.com/50888610/61229092-84fecc00-a6f5-11e9-9b12-e9e1caf5f0f0.gif), and then it reach a platue states. we also plotted the "score vs. ![CodeCogsEqn copy](https://user-images.githubusercontent.com/50888610/61229092-84fecc00-a6f5-11e9-9b12-e9e1caf5f0f0.gif)":
![alt text](Lscore.png)

we can see the Lasso regression can get a higher score when ![CodeCogsEqn copy](https://user-images.githubusercontent.com/50888610/61229092-84fecc00-a6f5-11e9-9b12-e9e1caf5f0f0.gif) is really small. at ![CodeCogsEqn copy](https://user-images.githubusercontent.com/50888610/61229092-84fecc00-a6f5-11e9-9b12-e9e1caf5f0f0.gif) = 0.0001, the MSE = 0.42578, and Test score = 0.39


##### Discussion 

|Model         | Linear        | Ridge,   ![CodeCogsEqn copy](https://user-images.githubusercontent.com/50888610/61229092-84fecc00-a6f5-11e9-9b12-e9e1caf5f0f0.gif) = 100| Lasso, ![CodeCogsEqn copy](https://user-images.githubusercontent.com/50888610/61229092-84fecc00-a6f5-11e9-9b12-e9e1caf5f0f0.gif)= 0.0001 | polynomial, n = 2 |
| ------------- | -------------| -------------------- |----------------------- |-------------------|
| MSE          | 0.5409        |0.3799.               | 0.4257                 | 0.4848            |
| TEST Score   | 0.299         |0.34                  | 0.39                   | 0.34            |


### Quality Prediction Using Logistic Regression
Logistic regression is applied on the training datatset (66%) of complete dataset. The mean accuracy from 3-fold cross validation of this model is 0.55. Classification performance of logstic classifier is shown in the confusion matrix. True positives for labels 5 and 6 are highest. Labels 3,4, 8, and 9 are not getting predicted or classified correctly. This could be because of lack of data points for these labels in the dataset, shown previously.

![alt text](logistic_confusionmatrix_tale.PNG)
![alt text](logistic_confusion_colored.PNG)


To check the consequences of overfitting, L2 norm regularized logistic regression model is developed. The accuracy of regularized logistic model on test data with varying value C (inverse of regualrization parameter) is plotted. With lower value of C that is higher penalty the accuracy is lower. As we increase the value of C the accuracy increases. However, it attains a plateau after reaching accuracy of around 0.55.

![alt text](performance_of_logistic.png)

Further, classification performance of logistic regression model on data in principal component space is tested. Training data is transformed to the principal component space that explains 99% of variance. Logistic regression model is fit on the transformed training dataset and applied to predict quality of the transformed test dataset. The resulting mean accuracy from 3-fold cross validation is 0.54. Thus, not much difference is observed in the accuracy obtained from using logistic model on dataset directly versus using logistic on dataset in principal componenet space. However, logstic regression classifier performed better than linear regression models discussed above.

### Quality Prediction Using Decision Tree and Random Ensemble Classifier
This section investigates performance of decision tree based classification models. 

#### Decision Tree Classifier
Decision tree classifier that uses entropy criterion to decide which feature should be splitted and at which value. The non-pruned decision tree is large (shown in figure below). Max depth of this decision tree is 23. 


![alt text](decision_tree_igone.svg)


   - The mean cross validated accuracy of the decision tree to predict labels is 0.55. Though this is similar to logistic model,
   impruned decision tree classifier takes time to perform large amount of computation. 
   - To perform pruning of the decision tree it is important to know either the imporant features or how much depth is sufficient to
   provide a good accuracy.
 
 Following graph presents variation in model accuracy with varying "max tree depth" parameter. 
 
![alt text](tree_Accuracy_versus_treedepth.PNG)

##### Example: Pre-Pruned Decision Tree Classifiers
Depth=5, Accuracy = 0.54

![alt text](prepruned_5.svg)

Depth=3, Accuracy = 0.52

![alt text](prepruned_3.svg)

Although a non-pruned decision tree based classifier for some cases is able to achieve accuracy of 0.59, the mean crossvalidated score is 0.55. To save computation time, the depth of tree can be reduced. With max depth of more than 15, the decision tree classifier performs better than logistic regression that gave accuracy of 0.55.

#### Ensemble Random Forest Classifier
To avoid overfitting issues from decision tree classifier and to investigate accuracy of prediction model, ensemble random forest classifier is fitted on training data. In ensemble random forest classifier, several samples of data are created by random sampling from dataset with replacement. Each sample is used to learn provide classification using decision trees (also known as estimators). The classification value obtained in majority among all trees is the classification that random forest classifier produces.

*Result for Random Forest Classifier*

1. n_estimators = 100
2. Depth_limit = None
3. Bootstrap = True, Model uses bootstrapped dataset to create trees instead of while dataset.

Accuracy score = 0.61

![alt text](rf_100_maxfeatures.svg)

1. n_estimators = 500
2. Depth_limit = None
3. Bootstrap = True, Model uses bootstrapped dataset to create trees instead of while dataset.

Accuracy score = 0.62

![alt text](rf_500_maxfeatures.svg)

Accuracy score increased significantly on using ensemble random forest classifier over using deicison tree classifier. 

### Quality Prediction Using Support Vector Machine

Support vector machine (SVM) was implemented for multi-class classification using "one against one" approach. Different kernel functions were applied such as RBF, polynomial and linear on both mixed (red and white wine data together) and separated datasets (separated as red and white wine). Datasets were splitted as 1/3 for test and 2/3 for training where standardization was applied only on training set.

While modelling the mixed dataset, red and white wine being a feature of dataset as the wine type, should be converted from string to an integer. This phenomenon has been investigated with RBF kernel. First, white wine was converted to 1 whereas red wine was 2. Also, using gridsearch with 3-fold cross validation, RBF kernel parameters of C and gamma were found. Confusion matrix for this case is shown below: 

![White_wine=1,red_wine=2](SVM-confusion_matrix-RBF-mixed_1.png)               

Then, same numbers were assigned vice versa (white wine = 2, red wine = 1) to observe any difference. Resulted confusion matrix is given below:

![White_wine=2,red_wine=1](SVM-confusion_matrix-RBF-mixed_2.png)

Much difference could not be observed. However, accuracy scores for both classifications were found as 64 %. Confusion matrices for white and red wine datasets on which 'RBF' kernel was implemented is shown below (Left-white wine, Right-red wine):

![White_wine, RBF](SVM-confusion_matrix-RBF-white.png)      ![Red_wine, RBF](SVM-confusion_matrix-RBF-red.png)

Accuracy scores for white and red wine datasets were found as 63 % and 60 %, respectively. Considering 64 % accuracy score for mixed dataset, a decreasing trend was observed for separated datasets. The reason may be relatively smaller training sets with separated datasets where the features are not varying much by the wine type.

After RBF, other kernel functions were also investigated. Below figure shows that RBF kernel outperforms among others in terms of the accuracy score, and is followed by polynomial and linear kernels, respectively.

![Accuracy_score, RBF](SVM-Accuracy_score.png) 

In conclusion, maximum accuracy was obtained with RBF kernel on mixed dataset as 64 %.  

## Conclusion

   - Prediction of wine quality for the given dataset was challenging because of the multi-class labeling (7 labels of quality)
   - The distribution of data for quality label instances in the dataset is imbalances. This might have made it diffcult for classifiers
   to learn to predict the labels for which data instances are less.
   - Principal component analysis on dataset revealed the "dense spread" of data in the principal component space.
   - Support vector machine classifier suited best for wine quality prediction, closely followed by Ensemble random forest classifier.
   - Next to them, Decision tree classifier and logistic regression classifier performed at par. 
   - Regularized linear regression classifiers (ridge regression and lasso regression) and polynomial linear regression model performed
   at par.
   - Simple linear regression classifier produced least accurate result.
   
Table below summarizes model accuracy values obtained for the 8 supervised learning algorithms investigated.

| Learning Classifier | Accuracy |
| ------------- | ------------- |
| Linear Regression  | 0.29 |
| Polynomial Regression  | 0.34 |
| Ridge Regression  | 0.34 |
| Lasso Regression  | 0.39 |
| Logistic Regression | 0.55 |
| Decision Tree  | 0.55 |
| Random Forest  | 0.61 |
| Support Vector Machine  | 0.64 |




## References
[1] P. Cortez, A. Cerdeira, F. Almeida, T. Matos, J. Reis, “Modeling wine preferences by data mining from physicochemical properties,” Decision Support Systems,” Vol. 47(4), 2009, p. 547-553.

[2] https://www.wineinstitute.org/resources/statistics/article86

[3] A. Legin, A. Rudnitskaya, L. Luvova, Y. Vlasov, C. Natale, and A. D’Amico. Evaluation of Italian wine by the electronic tongue: recognition, quantitative analysis and correlation with human sensory perception. Analytica Chimica Acta, pages 33–34, 2003.

[4] D. Smith and R. Margolskee. Making sense of taste. Scientific American, 284:26– 33, 2001.

## Thank you


