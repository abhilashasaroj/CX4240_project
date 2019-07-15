![alt text](https://www.nvtt.net/wp-content/uploads/2018/10/wine-tasting.jpg)

## Introduction
This study aims to compare the performance of different supervised learning algorithms to predict wine quality based on a wine’s physicochemical test results. Wine quality assessment is crucial in the wine industry and is assessed based on physicochemical tests at laboratories and on sensory tests that rely on human experts opinion on wine preferences [1].

## Background and Motivation - Renee
-> Why this is important? Can take from the proposal. 
-> give the ratio of test and training for all models 
-> Add outline of the webapage/presentation.

## Data Description and Initial Data Exploration
This section presents insights on the wine quality dataset which will be used to model quality predictions. 

### Data Distribution in Dataset by Quality Lables and Wine Type

Describe features, labels - Renee

Add visualizations - Renee
 
![alt text](pic1.JPG)

Present key insights - Renee

### Feature Pairplot
![alt text](feature_plots.png)

Key insight on correlation betwen features - Renee

## Initial Data Exploration 
### Principal Componenet Analysis - Abhilasha
A dataset with 12 feature pushes us to conduct a Principal Componenet Analysis on the data to consider possibility of dimenstion reduction. Principal componenet analysis result for complete dataset reveals that 0.99 of total variance in dataset is explained by first 11 eigen values. PCA result indicated not much of dimension reduction can be achieved by using 2 or 3 principal componenets. Thus, there are not easily identifiable directions in which signiifcant variance of the data points can be captured. 

[0.3176715, 0.21069888, 0.12999856, 0.08094213, 0.06034393, 0.0510322, 0.04471237, 0.04190673, 0.02934314, 0.02139346, 0.00994418]



![alt text](pc_2d.png)

<iframe width="700" height="700" frameborder="0" scrolling="no" src="//plot.ly/~abhilashasaroj/108.embed"></iframe>

Describe PC scores and result
Add visualizations

## Supervised Learning Models for Wine Quality Prediction
-> Add outline
### Quality Prediction Using Linear Regression and Polynomical Regression - Yi
First, let's look at the correlation among features and label('quality'):
![alt text](Cor-1.png)
From the correlation map, we can see that the most correlated feature with 'quality' is 'alchol'.
Then we want to plot the relation between each feature and label:
![alt text](LR%20in%20seaborn.png)
We can see that the linear relation bewteen each features and label is not very good prediction, so we want to do a linear combination of all features of dataset to do the prediction. 
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

##### Discussion 
By comparing the MSE between linear regression model and polynormial model, we can conclude that for our dataset, polynormial model (when n = 2) is a little bit better than linear regression model and n = other values.

### Quality Prediction Using Ridge Regression and Lasso Regression - Yi 
#### (3) Ridge Regression
For Ridge Regression, we add a regulation in the function to reduce the magnitude of the coefficients. Our goal is to find the 
![CodeCogsEqn copy](https://user-images.githubusercontent.com/50888610/61229092-84fecc00-a6f5-11e9-9b12-e9e1caf5f0f0.gif) that can optimize the parameters. ![CodeCogsEqn](https://user-images.githubusercontent.com/50888610/61228984-51bc3d00-a6f5-11e9-818e-9dd169da3edc.gif) If we look at the cofficients for different features in RR here:

![alt text](Rcof.png)
As we increase ![CodeCogsEqn copy](https://user-images.githubusercontent.com/50888610/61229092-84fecc00-a6f5-11e9-9b12-e9e1caf5f0f0.gif), we will see that the magnitude of coefficeints decrease.This is because higher the ![CodeCogsEqn copy](https://user-images.githubusercontent.com/50888610/61229092-84fecc00-a6f5-11e9-9b12-e9e1caf5f0f0.gif)  is, bigger is the penalty and therefore, the magnitude of coefficients are reduced.

if we look at the "MSE vs. ![CodeCogsEqn copy](https://user-images.githubusercontent.com/50888610/61229092-84fecc00-a6f5-11e9-9b12-e9e1caf5f0f0.gif)" :
![alt text](RMSE1.png)
 we can see that MSE will decrease first as ![CodeCogsEqn copy](https://user-images.githubusercontent.com/50888610/61229092-84fecc00-a6f5-11e9-9b12-e9e1caf5f0f0.gif) increases and then it ramp to really high value,the lowest MSE was reached when ![CodeCogsEqn copy](https://user-images.githubusercontent.com/50888610/61229092-84fecc00-a6f5-11e9-9b12-e9e1caf5f0f0.gif)is 100, we also plotted the "score vs. ![CodeCogsEqn copy](https://user-images.githubusercontent.com/50888610/61229092-84fecc00-a6f5-11e9-9b12-e9e1caf5f0f0.gif)":
![alt text](Rscore1.png)

we see for this case, score is presentinf an opposite trend as MSE, but it also shows a highest score at ![CodeCogsEqn copy](https://user-images.githubusercontent.com/50888610/61229092-84fecc00-a6f5-11e9-9b12-e9e1caf5f0f0.gif) = 100.

##### at ![CodeCogsEqn copy](https://user-images.githubusercontent.com/50888610/61229092-84fecc00-a6f5-11e9-9b12-e9e1caf5f0f0.gif)a = 100, the MSE = 0.3799 and the Test score = 0.34

#### (4) Lasso Regression

For Lasso Regerssion, ![CodeCogsEqn-2](https://user-images.githubusercontent.com/50888610/61230953-0c9a0a00-a6f9-11e9-9cda-8a1c3f04beff.gif), the regulation method is different with Ridge Regression. in RR, the regulation is related to squared coefficience, however, in LR, the regulation is only related to absolute value of coefficience. Therefore, we are expecting that when we increase the value of ![CodeCogsEqn copy](https://user-images.githubusercontent.com/50888610/61229092-84fecc00-a6f5-11e9-9b12-e9e1caf5f0f0.gif), coefficients are approaching towards 0!!!! therefore, Lasso selects some features while reduce the coefficients of others to zero. if we look at the the cofficients for different features in LR here:
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
| TEST Score   | 0.299         |0.34                  | 0.39                   | 0.299             |


### Quality Prediction Using Logistic Regression - Abhilasha

![alt text](logistic_confusionmatrix_tale.PNG)

![alt text](logistic_confusion_colored.png)


![alt text](performance_of_logistic.png)

#### Discussion 

### Quality Prediction Using Decision Tree and Random Ensemble Classifier - Abhilasha
#### Decision Tree Classifier

![alt text](Project_saroj.svg)

Accuracy = 

##### Pruned Decision Tree Classifier
Depth=5
![alt text](Project_saroj_5.svg)

Accuracy =

Depth =3
![alt text](Project_saroj_3.svg)

Accuracy =

##### Discussion 

#### Ensemble Random Forest Classifier

Result for Random Forest Classifier with following attributes

1. n_estimators = 100
2. Depth_limit = None
3. Bootstrap = True, Model uses bootstrapped dataset to create trees instead of while dataset.

*Accuracy score = 0.62*

![alt text](Project_saroj_rf_estimator2_n100_ootstrap.svg)


### Quality Prediction Using Support Vector Machine - Ogulcan

Support vector machine (SVM) was implemented for multi-class classification using "one against one" approach. Different kernel functions were applied such as RBF, polynomial and linear on both mixed (red and white wine data together) and separated datasets (separated as red and white wine). Datasets were splitted as 1/3 for test and 2/3 for training where standardization was applied only on training set.

While modelling the mixed dataset, red and white wine being a feature of dataset as the wine type, should be converted from string to an integer. This phenomenon has been investigated with RBF kernel. First, white wine was converted to 1 whereas red wine was 2. Also, using gridsearch with 3-fold cross validation, RBF kernel parameters of C and gamma were found. Confusion matrix for this case is shown below: 

![White_wine=1,red_wine=2](SVM-confusion_matrix-RBF-mixed_1.png)               

Then, same numbers were assigned vice versa (white wine = 2, red wine = 1) to observe any difference. Resulted confusion matrix is given below:

![White_wine=2,red_wine=1](SVM-confusion_matrix-RBF-mixed_2.png)

Much difference could not be observed. However, accuracy scores for both classifications were found as 0.64. Confusion matrices for white and red wine datasets on which 'RBF' kernel was implemented is shown below (Left-white wine, Right-red wine):

![White_wine, RBF](SVM-confusion_matrix-RBF-white.png)      ![Red_wine, RBF](SVM-confusion_matrix-RBF-red.png)

Accuracy scores for white and red wine datasets were found as 0.63 and 0.60, respectively. Considering 0.64 accuracy score for mixed dataset, a decreasing trend was observed for separated datasets. The reason may be relatively smaller training sets with separated datasets where the features are not varying much by the wine type.

After RBF, other kernel functions were also investigated. Below figure shows that RBF kernel outperforms among others in terms of accuracy score, and is followed by polynomial and linear kernels, respectively.

![Accuracy_score, RBF](SVM-Accuracy_score.png) 

In conclusion, maximum accuracy was obtained with RBF kernel on mixed dataset as 64 %.  

#### Model Cross Validation Results

#### Discussion 

## Comparison of Prediction Models

| First Header  | Second Header |
| ------------- | ------------- |
| Content Cell  | Content Cell  |
| Content Cell  | Content Cell  |

## Conclusion

## References

Thank you


