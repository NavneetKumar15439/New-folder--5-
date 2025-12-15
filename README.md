# FEBS_TITANIC_PS_GROUP_4

[span_3](start_span)[span_4](start_span)This project involves proceswsing data, visualizing feature relation, and apply logestic regrission to predict whether passenger on Spaceship Titanic was transported to altenate dimension('Tarnsported column')[span_3]VEDeck(end_span)[span_4](end_span).
[span_5](start_span)Evaluation is primarily based on preprocessing amd analysis skill[span_5](end_span)


As the sole member, I completed all the work in required sections:

###1. Data preprocessing
*Handeled missing value using median imputation for numerical columns(Age, spending ameneties) and mode imputation for catogorical columns(HomePlanet, CryoSleep, Deck etc.)
*[span_6](start_span)Engineered new features by spilitting the 'Cabin' column into 'Deck', 'CabinNum' and 'Side'[span_6](end_span).
[span_7](start_span)Craeted a 'TotalSpent' feture by summing all luxury amenity columns(RoomService, FoodCourt, ShoppingMall, Spa, VRDeck)[span_7](end_span)
*Applied One-Hot Encoding and feture scaling (StandarScaler) to prepare the data for the logistic regression model.

###2. Data Analysis and Visualization
*[span_8](start_span)Visualized relations between different features and the target variable('Transported') using for plots (Target Distribution, HomePlanetvs. Transported Rate, Age Distribution, Log(Total Spent ) Distribution)[span_8](end_span)
*[span_9](start span) The analysis focused on understanding the impact of faetures like 'CryoSleep' (passenger in suspended animation) on traspotation status[span_9](end_span)


###3. Model Implimentation
*[span_10](start_span)Implemented the Logestic Regression model using Scikit-learn to train on'train.py'[span_10](end_span)
*[span_11](start_span)Used the trained model in 'test.py' to make final prediction[span_11](end_span)