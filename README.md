# PTI Hack 2022
Track №3 task solution by Team GARCH

## Task
Competition could be found by this link: https://www.kaggle.com/c/pti-hack <br />
<br />
The task was to predict the probability of successful deal closing, having the history of interactions with clients.



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
