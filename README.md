# Heart-Disease-Classification



This notebook uses various data science librabries in attempt to build a machine learning model capable of detecting whether a patient has heart disease, based on their medical attributes.

We've taken the following appraoch:

1. Problem definition
2. Data
3. Evaluation
4. Features
5. Modelling
6. Conclusion

## 1. Problem definition
Binary Classification: Predict whether or not a patient has heart disease, based on some of their medical attributes

## 2. Data

This Cleveland database is used by machine learning researchers and originally published by UCI Machine Learning Repository: https://archive.ics.uci.edu/ml/datasets/Heart+Disease
    
A similar version of this database can be found on Kaggle (https://www.kaggle.com/ronitf/heart-disease-uci) however, there are some important differences between the UCI and Kaggle versions of the data. This is addressed in the Features section.

**Credits:**

1. Hungarian Institute of Cardiology. Budapest: Andras Janosi, M.D.
2. University Hospital, Zurich, Switzerland: William Steinbrunn, M.D.
3. University Hospital, Basel, Switzerland: Matthias Pfisterer, M.D.
4. V.A. Medical Center, Long Beach and Cleveland Clinic Foundation: Robert Detrano, M.D., Ph.D.

**Donor:**

David W. Aha (aha@ics.uci.edu) (714) 856-8779

**Data Dictionary (original)**
1. age - age in years
2. sex - gender (1 = male; 0 = female)
3. cp - chest pain type
    * 0 = Typical angina: chest pain related decrease blood supply to the heart
    * 1 = Atypical angina: chest pain not related to heart
    * 2 = Non-anginal pain: typically esophageal spasms (non heart related)
    * 3 = Asymptomatic: chest pain not showing signs of disease
4. trestbps - resting blood pressure (in mm Hg on admission) anything above 130-140 is typically cause for concern
5. chol - serum cholestoral in mg/dl
    * serum = LDL + HDL + .2 * triglycerides
    * above 200 is cause for concern
6. fbs - (fasting blood sugar > 120 mg/dl) (1 = true; 0 = false)
    * '>126' mg/dL signals diabetes
7. restecg - resting electrocardiographic results
    * 0 = Nothing to note
    * 1 = ST-T Wave abnormality
        * can range from mild symptoms to severe problems
        * signals non-normal heart beat
    * 2 = Possible or definite left ventricular hypertrophy
        * Enlarged heart's main pumping chamber
8. thalach - maximum heart rate achieved
9. exang - exercise induced angina (1 = yes; 0 = no)
10. oldpeak - ST depression induced by exercise relative to rest looks at stress of heart during excercise unhealthy heart will stress more
11. slope - the slope of the peak exercise ST segment
    * 0 = Upsloping: better heart rate with excercise (uncommon)
    * 1 = Flatsloping: minimal change (typical healthy heart)
    * 2 = Downslopins: signs of unhealthy heart
12. ca - number of major vessels (0-3) colored by flourosopy
    * colored vessel means the doctor can see the blood passing through
    * the more blood movement the better (no clots)
13. thal - thalium stress result
    * 0 - 3 = normal
    * 6 = fixed defect: used to be defect but ok now
    * 7 = reversable defect: no proper blood movement when excercising
14. target - has disease or not (1=true, 0=false)

## 3. Evaluation

Due to the significance of the outcome of the prediction,
> We intend to pursue the project if we can achieve 95% accuracy, in our predictions. 

## 4 Features

**Data Quality**
The below-mentioned changes were made to address the versioning descrepancies based on Kaggle discussions (https://www.kaggle.com/ronitf/heart-disease-uci/discussion/105877):

1. cp: chest pain type (0 = asymptomatic, 1 = atypical angina, 2 = non-anginal pain, 3 = typical angina)
2. restecg: resting electrocardiographic results (0 = showing probable or definite left ventricular hypertrophy, 1 = normal, 2 = having ST-T wave abnormality).
3. slope: the slope of the peak exercise ST segment (0 = downsloping, 1 = flat, 2 = upsloping)
4. target: heart disease present (0 = heart disease, 1 = no heart disease)
5. thal: thalium stress result (0 = NaN, 1 = fixed defect, 2 = normal, 3 = reversable defect)
    * Remove NaN (thal = 0)
6. ca - number of major vessels colored by flourosopy (0 = 0, 1 = 1, 2 = 2, 3 = 3, 4 = NaN)
    * Remove NaN (ca = 4)

**Data Dictionary (updated)**
1. age - age in years
2. sex - gender (1 = male; 0 = female)
3. cp - chest pain type
    * 0 = Asymptomatic: chest pain not showing signs of disease
    * 1 = Atypical angina: chest pain not related to heart
    * 2 = Non-anginal pain: typically esophageal spasms (non heart related)
    * 3 = Typical angina: chest pain related decrease blood supply to the heart
4. trestbps - resting blood pressure (in mm Hg on admission) anything above 130-140 is typically cause for concern
5. chol - serum cholestoral in mg/dl
    * serum = LDL + HDL + .2 * triglycerides
    * above 200 is cause for concern
6. fbs - (fasting blood sugar > 120 mg/dl) (1 = true; 0 = false)
    * '>126' mg/dL signals diabetes
7. restecg - resting electrocardiographic results
    * 0 = Possible or definite left ventricular hypertrophy
        * Enlarged heart's main pumping chamber
    * 1 = Normal, with nothing to note
    * 2 = ST-T Wave abnormality
        * can range from mild symptoms to severe problems
        * signals non-normal heart beat
8. thalach - maximum heart rate achieved
9. exang - exercise induced angina (1 = yes; 0 = no)
10. oldpeak - ST depression induced by exercise relative to rest looks at stress of heart during excercise unhealthy heart will stress more
11. slope - the slope of the peak exercise ST segment
    * 0 = Downsloping: signs of unhealthy heart
    * 1 = Flat: minimal change (typical healthy heart)
    * 2 = Upsloping: better heart rate with excercise (uncommon)
12. ca - number of major vessels (0-3) colored by flourosopy
    * colored vessel means the doctor can see the blood passing through
    * the more blood movement the better (no clots)
13. thal - thalium stress result
    * 0 = NaN (to be removed)
    * 1 = fixed defect: used to be defect but ok now
    * 2 = normal
    * 3 = reversable defect: no proper blood movement when excercising
14. target - has disease or not (0=true, 1=false)

## 5. Modelling

There are some interesting high-level correlations features, which we'll explore more using these classifcation models:

Logistic Regression (Classifier)
K-Nearest Neighbors Classifier
Random Forest Classifier
Decision Tree Classifier
Support Vector Classifier
Gradient Boost Classifier

Baseline models produced decent accuracy results, some of which were improved further through the tuning of hyperparameters.

## 6. Conclusion

Unfortunately, we would not able to accurately predict whether a patient has heart disease or not, based on the below: 
* General results are not complimentary nor intuitive. 
* The maximum cross-validated Accuracy score is 86.7%, which is lower than the acceptable target.
* Data-related issues such as class-imbalance and size of the dataset needs to be addressed before further exploration. 
