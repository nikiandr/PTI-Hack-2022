# PTI Hack 2022
Track №3 task solution by Team GARCH with RMSE 20.12592

## Task
Competition could be found by this link: https://www.kaggle.com/c/pti-hack <br />
<br />
The task was to predict the probability of successful deal closing, having the history of interactions with clients.
<br />
  - - - -
## Solution

Our solution approach consists of building a Classifier, then using LightGBM and CatBoost separately for successful and unsuccessful cases and stacking them. Since the train and test data had a non-zero intersection, we had to define a correct time-aware prediction scheme (so that we could predict the target for each new element based only on past data). 
Markup : ![picture alt](https://imgur.com/a/2CdmSGn "prediction scheme")
<br />
Reasons for using this method:
 * Avoid overfitting at the intersection of train and test
 * Avoid occuring leaks during the generation of new features (Situation where the past flows into the future)
<br />

##  Feature Generation
1) Dates: 'CreatedDate', 'CreatedDateForInsert', 'ValidThroughDate', differences, quarters, years, sin-cos encoding
2) Lags: Stats of previous probabilities grouped by Opportunity, CreatedBy and periods
3) Categorical: 'CreatedById', 'AccountId', 'RecordTypeId', 'Type', 'LeadSource', 'CampaignId' etc.
4) Dividing feature "Needs__c" using CountVectorizer.
<br />

## Approach Steps

 * Divide the target by 100 and build the LightGBM model with the loss function. <br />
 * Build a classifier model (target - "StageName" - forecast of how the deal will end at the very end: 0 - unsuccessfully, 1 - successfully). <br />
 * For each point in the dataset, predict the value <br />
 * Divide the dataset into 2 parts: successful and unsuccessful cases.   <br />
 * On each of the parts, build a separate LGBMRegressor and CatBoost to predict the final value of the probability.<br />
 * Stacking of CatBoost and LightGBM models (with coefficients 0.4 and 0.6, respectively) in each of the categories.<br />
<br />
  - - - -

## Notebooks
- `pti_hack.ipynb` - Main notebook used for creating the final stacking model

Because of exhaustive pointwise time-respecting predictions for stacking - the notebook takes approximately 1 hour to run on 16 CPUs / n_jobs=32.

## Docker
The same code in .py script and additional files to run within the Docker container

### Build
```
cd PTI-Hack-2022/docker
docker build -t pti_hack .
docker tag pti_hack:latest <your_username>/pti_hack:latest
docker push <your_username>/pti_hack
```

### Environmental variables
```
KAGGLE_USERNAME - username in Kaggle
KAGGLE_TOKEN - Kaggle API token
N_JOBS - number of jobs for parallel execution
```

### Running on zod.tv (sponsor platform)
```
curl -H "Content-Type: application/json" \
-H "Authorization: Zod58 {{your_api_key}}" \
-X POST https://offchain.zod.tv/job_new -d @- << EOF
{
    "type": "docker",
    "path": "docker.io/<your_username>/pti_hack",
    "cpu": 32,
    "ram": 64,
    "disk": 30
}
EOF
```

The prediction will be submitted automatically after execution ends.

### Troubleshooting
For some reasons container sometimes fails with the `joblib.externals.loky.process_executor.TerminatedWorkerError` when running on zod.tv

Consider `N_JOBS` to be small enough to prevent this (but this can significantly slow down the learning speed).
