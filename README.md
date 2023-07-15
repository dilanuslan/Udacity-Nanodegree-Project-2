# Disaster Response Pipeline Project

### Summary of The Project
For the second project of Udacity's Data Scientist Nanodegree program, I analyzed the disaster data provided by Appen (formerly Figure Eight) which includes real messages that were sent during disasters. These messages had to be categorized in order to provide the appropriate help. Therefore, I created a machine-learning pipeline and categorized the messages. The project also includes a web interface and you can add new messages to classify accordingly. The index page will show you the 36 categories that were identified. 

### File Tree
<img width="562" alt="Ekran Resmi 2023-07-15 17 43 45" src="https://github.com/dilanuslan/Udacity-Nanodegree-Project-2/assets/32961512/402f4dad-4786-49c4-9d03-b0b6591e9fa8">

### Project Detail
This project includes 3 main parts: ETL pipeline, ML pipeline, Web App. I will give some further details in this section. 

#### 1. ETL Pipeline
ETL pipeline was implemented in the data/process_data.py script and it has the following steps:
- Loads the datasets provided by Appen (messages and categories)
- Merges the two datasets and creates new columns
- Cleans the data
- Store the results in a database (SQLite in this project)

#### 2. ML Pipeline
ML pipeline was implemented in the models/train_classifier.py script and it has the following steps:
- Loads the data from the SQLite database
- Splits the data into test and training sets
- Builds a pipeline that applies text processing, TFIDF transformation and Multi Output Classification
- Train and fit a model using GridSearchCV and choosing the best parameters
- Outputs the results on the test set
- Produces a pickle file

#### 3. Web App
The project contains a web interface to see the categories and classify the messages. Screenshots of the project are provided below:

<img width="827" alt="Ekran Resmi 2023-07-15 18 35 44" src="https://github.com/dilanuslan/Udacity-Nanodegree-Project-2/assets/32961512/f02f7eea-91af-4614-b3c5-4dadbf7172e3">

<img width="827" alt="Ekran Resmi 2023-07-15 18 37 29" src="https://github.com/dilanuslan/Udacity-Nanodegree-Project-2/assets/32961512/a3736d69-67f0-4ba4-8886-ec7a04502688">

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

### Acknowledgements
This project was built with the help of Udacity instructions.
