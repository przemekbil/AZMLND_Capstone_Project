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

Because I was trying to solve a classification problem (1-batch has failing parts or 0-batch has all good parts) I choosed to use LogisticRegression algorithm from sklearn library. 2 parameters have been adjusted through HyperDrive: inverse of regularization strength (parameter C) and Maximum number of iterations taken for the solver to converge (max_iter).

The larger the regularization strength (smaller values of C parameter) the higher the penalty for increasing the magnitude of parameters. This is to prevent overfitting the model to the train data and make it more general i.e. also applicable to the unseen test data.

There are 2 parameters sweeping modes available in Azure ML: Entire grid and Random Sweep. Entire grid sweep is exhaustive, tends to give slightly better results but is time consuming. Random sweep in contrast offers good results without taking as much time. For this project, I'm going for the Random sweep using RandomParameterSampler Class using the following search spaces:

* max_iter parameter will be selected out of 4 predefined values (50, 100, 200, or 400)
* C parameter will be randomly pooled from the interval between exp(-10) to exp(10) using logunifrom distribution. Since the parameter C represents the inverse of the regularization strength, logunifrom distribution has been selected to attain distribution of regularization strength as close to uniform as possible.

For early stopping policy, I selected a BanditPolicy class. This policy compares the current training run with the best performing run and terminates it if it’s performance metric drops below calculated threshold. The main benefit of using this policy comparing to the other 2 policies is that the current runs are terminated after comparing with the best performing run. If the current run performance drops greatly below the best run's performance, it will be terminated. 

The parameters I used in my experiment:
```
policy = BanditPolicy(slack_factor=0.05, evaluation_interval=5, delay_evaluation=10)
```
would cause each run to be compared with the best performing run after each 5 algorithm runs (starting after first 10 runs) and if the run’s performance drops below 95% of current best run performance, then it would get terminated.

### Results
*TODO*: What are the results you got with your model? What were the parameters of the model? How could you have improved it?

The best run was achieved with C= 60.8728 and max_iter=100 and the resulting model have reported Accuracy of 0.676:


![image](https://user-images.githubusercontent.com/77756713/138364877-faf83d3f-4aaa-436f-aa87-1797f608ce99.png)


I have also run an experiment with a BayesianParameterSampling instead of RandomParameterSampling and the result were similar, the best performing model had Accuracy of 0.670, which is just below the accuracy of the previous experiment:

![image](https://user-images.githubusercontent.com/77756713/140111446-26294964-f63f-4030-8c2e-52b4e73a0f47.png)


## Model Deployment

The model with the highest Accuarcy came from the Hyperparameter Tuning run and its reported Accuracy is 0.676. As the best performing model, this model has been selected for deployment. It got deployed as an endpoint 'sinter-scrap-model2':

![image](https://user-images.githubusercontent.com/77756713/140139583-f7a1781a-427d-40ec-9b0c-1f71775986cb.png)

To query the endpoint, the input data has to have the same format as in the [Sample_input file](Sample_input) and it has to contain all the 83 columns listed in this file. The best way to create sample input dataset like that is to apply the follwoing operations on the raw production data:

```
raw_data = pd.read_csv('./data/balanced.csv')
raw_data = raw_data.dropna()
raw_data = raw_data.drop("WO-MRR NUM", axis=1)
raw_data = raw_data.drop("Has MRR", axis=1)
raw_data = pd.get_dummies(raw_data, columns=['PROD_CDE','PREP ASSOCIATE', 'FURN'])
```


## Screen Recording

Below is the link to the screencast recording: 
https://www.youtube.com/watch?v=p2hWM2vxlGw

It contins a brief overview of the following:
- A working model
- Demo of the deployed  model
- Demo of a sample request sent to the endpoint and its response

