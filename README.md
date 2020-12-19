# Airplane Accident Severity Prediction
Flying has been the go-to mode of travel for years now; it is time-saving, affordable, and extremely convenient. According to the FAA, 2,781,971 passengers fly every day in the US, as in June 2019. Passengers reckon that flying is very safe, considering strict inspections are conducted and security measures are taken to avoid and/or mitigate any mishappenings. However, there remain a few chances of unfortunate incidents.

## tl;dr
* Compared different ensemble learning models to find the best fit for the data
* Created a flask application using HTML and Bootsrap to deploy the Machine Learning model


## Technologies Used
    
* <strong>Python</strong>
* <strong>Pandas</strong>
* <strong>Numpy</strong>
* <strong>Matplotlib</strong>
* <strong>Seaborn</strong>
* <strong>Scikit Learn</strong>
* <strong>Ensemble Learning</strong>
* <strong>Flask</strong>
* <strong>Bootstrap</strong>
* <strong>HTML</strong>
* <strong>CSS</strong>

## Data
The dataset consists of certain parameters recorded during the incident⁠ such as cabin temperature, turbulence experienced, number of safety complaints prior to the accident, number of days since the last inspection was conducted before the incident, an estimation of the pilot’s control given the various factors at play, and the likes. 

### Pairplot of the data
<p align="center">
  <img src="https://github.com/chawla201/Airplane-Accident-Severity-Prediction/blob/master/images/pairplot.png" width=600>
</p>

## Results
|Classifier | With Safety Score (accuracy %) | Without Safety Score (accuracy %) |
| --- | --- | --- |
|Decision Tree Classifier |  94.54 | 53.73 |
|Random Forest Classifier |  89.92 | 66.60 |
|Gradient Boosting Classifier |  96.87 | 64.80 |
|Adaptive Boosting Classifier |  94.94 | 64.93 |
|XGBoost Classifier |  96.4 | 70.07 |

## Model Deployment
Developed a simple web application using <strong>Flask</strong> and <strong>Bootstrap</strong> to deploy the Gradient Boosting Classifier

## Screenshots
<p align="center">
  <img src="https://github.com/chawla201/Airplane-Accident-Severity-Prediction/blob/master/images/home_page.png" width=600>
  <img src="https://github.com/chawla201/Airplane-Accident-Severity-Prediction/blob/master/images/prediction.png" width=600>
</p>
