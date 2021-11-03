# UDACITY Machine Learning Engineer with Microsoft Azure - Capstone project

This is the final project for my Machine Learning Engineer with Microsoft Azure course. In this project I have used both hyperdrive and automl modules to trin a classification model on a dataset sourced from the production data in TiPorocoat Value Stream. The classification models are trying to predict if there is going to be failig parts in the production batch based on the processing inormation.

## Project Set Up and Installation

To make sure project runs correctly in the Azure ML Studio environment, create folders 'source_dir' and 'data'. Copy datafile 'balanced.csv' into the 'data' folder and both Python scripts into the 'source_dir' folder as per the screenshot below:

![image](https://user-images.githubusercontent.com/77756713/139918574-ef68310a-0d88-4c24-83f1-e6eb69748598.png)


## Dataset

### Overview

In the project, I have used production data from the TiPorocoat value stream downloaded from our MES (Manufacturing Execution System). This data was taken from this years live production records (WK1-WK42 of 2021).

The production process in TiPorocoat value stream consists of 4 major steps: 

![image](https://user-images.githubusercontent.com/77756713/140050169-524ca5f5-c874-4df9-9be0-9ec9a508cf0d.png)

First, parts are blasted with dry media to create a rough surface that will help binding agent to stick to the parts. Next step is the most important one, Manual preparation. In this step, Associates manually apply binding agent to the parts and then apply thin layer of Titatium beads. Parts are then moved to the Sintering Furnace, where the binding agent evaporates and the beads under the high temperature are fused with the parts. The last step is the Visual Ispection, where parts are inspected for the quality of the beads coating.

It has been well known, that because of the manual process of the coating preparation, the biggest influence on the quality of the coating (and the scrap ratio after Visual Inspection) had Associates technique and experience. Parts prepared by some of the Associates tend to have better quality coating than the others. 

### Task

The aim of this project is to trying to predict which production batches (identified by the WO-MRR NUM) will have parts failing Visual Inspection step (HAS SCRAP AT SINTER=1) and which ones will have all parts passing the inspection (HAS SCRAP AT SINTER = 0). I will use 14 below independent variables to try to achieve that:

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
|Prep Week| 1-42| Number of week at which parts were manually prepped|
|FURN| String| Identifier of the Asset at which sintering process took place|
|HAS SCRAP AT SINTER|0,1 | Identifier if batch has failing parts after sinter process - the variable we are trying to predict| 




### Access

I have uploaded my data manually as a csv file into the 'data' dolder.

For the use in the AutoML module, I have completed all the data cleanup and preparation inside my automl notebook and then I have registered resulting Pandas dataframe as a Dataset called 'Sinter' withing the Azure environment. The AutoML run is accessing the 'Sinter' Dataset directly.

For the use in the Hyperparameter Tuning module, I have completed all the data cleanup and preperation inside the 'train.py' script. There was no need to register it as a Dataset as all the data was accessed direcltly from the uploaded csv file.

## Automated ML

In the AutoML run, I have used the following settings:

```
automl_settings = {
    "experiment_timeout_minutes": 40,
    "max_concurrent_iterations": 2,
    "primary_metric" : 'accuracy'
}
```

```experiment_timeout_minutes``` - It defines in minutes the experiment run time. I have run the experiment with minutes set to 20 and then to 40. There was no measuremble improvement in the eccuracy of the best model.

```max_concurrent_iterations``` - Represents the maximum number of iterations that would be executed in parallel. The computer cluster used in this experiment have 2 nodes, 1 experiment per node can be run concurently so the ```max_concurrent_iterations``` was set to 2

```primary_metric``` - the metric that AutoML will optimize for the model selection. I selected the same metric as in the hyperparameter tuning method: accuracy.



And the following configuration:

```
automl_config = AutoMLConfig(compute_target=cpu_cluster,
                             task = "classification",
                             training_data=dataset,
                             label_column_name="HAS SCRAP AT SINTER",
                             enable_early_stopping= True,
                             featurization= 'auto',
                             debug_log = "automl_errors.log",
                             **automl_settings
                            )
```



### Results

The model with the best accuracy was obtained using VoitingEnsemble Algorithm and it's accuracy was 0.64234:

![image](https://user-images.githubusercontent.com/77756713/139940126-ae578f6e-a398-4462-b924-2439cebff859.png)

![image](https://user-images.githubusercontent.com/77756713/138364289-81f7a2c3-8fd7-4a85-ba16-b991ad19d9b6.png)

The VotingEnsemble was constructed out of previous best performing unique runs, with individual weights assigned to each of them to optimize the overall performance. The inner estimators, their weights, Iteration number and original metrics are listed in the table below:

ITERATION | Algorithm | Weight | Metric |
--- | --- | --- | --- |
13 | maxabsscaler, sgdclassifierwrapper | 0.3 | 0.6355 | 
7 | maxabsscaler,	logisticregression |	0.1 | 0.6329|
12|	maxabsscaler,	logisticregression|	0.1|0.6329|
16|	standardscalerwrapper,	logisticregression|	0.1|0.6309	|
20|	truncatedsvdwrapper,	randomforestclassifier|	0.1|0.6295	|
0|	maxabsscaler,	lightgbmclassifier|	0.1|0.6225	|
11|	standardscalerwrapper,	xgboostclassifier|	0.1|0.6074	|
31|	maxabsscaler,	extratreesclassifier|	0.1|0.5555	|


According to the best performing model, top 4 features (columns) with the largest importance were: Prep week, Quantity prepared in the shift, Prep hour adn prep quantity:
![image](https://user-images.githubusercontent.com/77756713/140066576-4b8c528e-3d45-4d77-b5b5-c83c16393c6b.png)


The confusion table for this model is presented below:

![image](https://user-images.githubusercontent.com/77756713/140067200-1d585c6f-4eff-4de6-8bf9-ddd0940b6aee.png)

Although the models accuracy is moderate 64%, at least it seems to be well balanced.

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
