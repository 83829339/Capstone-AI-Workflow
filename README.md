# Capstone-AI-Workflow
Capstone AI Workflow

I ve included all the files for my AI Enterprise Workflow Capstone project.

Files included:

Directories included:

data/cs-train: all provided data to train the model

models: some trained models and tested models saved for prediction

logs: logs created during test-training-prediction

templates and static directories: files (html and additional static content) needed for rendering the flask application

unittests: 3 unit tests (for API,model and logs) that can be run executing capstoneRun-tests.py

app.py: Flask app to use call train API and predict API 

capstoneDataVisualization.py: Data ingestion, processing and visualization

capstoneModelBuildingSelection: Building, training, loading and predicting models

capstoneMonitoring.py: monitoring performance

capstoneRun-tests.py: to run the 3 unit tests

Dockerfile: file to execute app.py on a container after building the docker image (Capstone-AI-Workflow) and running the container on port 80

logger.py: to create and append info to the different logs

req.txt: python libraries needed to be installed in the container 
