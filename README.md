# UDACITY Machine Learning Engineer with Microsoft Azure - Capstone project

This is the final project for my Machine Learning Engineer with Microsoft Azure course. In this project I have used both hyperdrive and automl modules to trin a classification model on a dataset sourced from the production data in TiPorocoat Value Stream. The classification models are trying to predict if there is going to be failig parts in the production batch based on the processing inormation.

## Project Set Up and Installation

To make sure project runs correctly in the Azure ML Studio environment, create folders 'source_dir' and 'data'. Copy datafile 'balanced.csv' into the 'data' folder and both Python scripts into the 'source_dir' folder as per the screenshot below:

![image](https://user-images.githubusercontent.com/77756713/139918574-ef68310a-0d88-4c24-83f1-e6eb69748598.png)


## Dataset

### Overview

In the project, I have used production data from the TiPorocoat value stream downloaded from our MES (Manufacturing Execution System). The data for training was taken from this year to date (WK1-WK42 of 2021).

| Column name | Data type |column Explanation|
|---------|--|------------|
|WO-MRR NUM| Integer |Unique production batch idntifier|
|IS REWORK| 0,1| identifier wheter or not batch goes through processing fo the second time|
|PROD_CDE| String| Product code|
|HOURS BLAST TO PREP | Integer |Number of hours that passed from Blast operation until Manual preparation|
|PREP DAY |1-7| Day of week at which batch was manually prepared|
|PREP SHIFT|1,2| Shift at which batch was prepared|
|PREP H |0-23| Hour at which batch was prepared|
|PREP QTY| Integer| Numer of parts in the batch at manual preparation step |
|PREP ASSOCIATE | Integer| Unique identifier of an Associate preparing the batch |
|PREP QTY IN THE SHIFT | Integer| Number of parts prepared by associate in the current shift|
|PREPED IN THE SHIFT SO FAR | integer | Number of parts prepared by an Associate in the shift prior to current batch |
|HOURS PREP TO FUR	| integer| Number of hours that passed from prep operation until the Furnace operation|
|ASSOCIATE EXPERIENCE| 0-3| Associates experience in manual preparation based on amount of parts prepared so far|
|Prep Week| 1-43| Number of week at which parts were manually prepped|


### Task
*TODO*: Explain the task you are going to be solving with this dataset and the features you will be using for it.

### Access
*TODO*: Explain how you are accessing the data in your workspace.

## Automated ML
*TODO*: Give an overview of the `automl` settings and configuration you used for this experiment




### Results
*TODO*: What are the results you got with your automated ML model? What were the parameters of the model? How could you have improved it?



*TODO* Remeber to provide screenshots of the `RunDetails` widget as well as a screenshot of the best model trained with it's parameters.
![image](https://user-images.githubusercontent.com/77756713/138364289-81f7a2c3-8fd7-4a85-ba16-b991ad19d9b6.png)


![image](https://user-images.githubusercontent.com/77756713/138364402-82736541-11d9-46f5-a320-dbbb82bdac2f.png)

## Hyperparameter Tuning
*TODO*: What kind of model did you choose for this experiment and why? Give an overview of the types of parameters and their ranges used for the hyperparameter search


### Results
*TODO*: What are the results you got with your model? What were the parameters of the model? How could you have improved it?

*TODO* Remeber to provide screenshots of the `RunDetails` widget as well as a screenshot of the best model trained with it's parameters.

![image](https://user-images.githubusercontent.com/77756713/138364877-faf83d3f-4aaa-436f-aa87-1797f608ce99.png)


## Model Deployment
*TODO*: Give an overview of the deployed model and instructions on how to query the endpoint with a sample input.

## Screen Recording

Below is the link to the screencast recording: 
https://www.youtube.com/watch?v=p2hWM2vxlGw

It contins a brief overview of the following:
- A working model
- Demo of the deployed  model
- Demo of a sample request sent to the endpoint and its response

## Standout Suggestions
*TODO (Optional):* This is where you can provide information about any standout suggestions that you have attempted.
