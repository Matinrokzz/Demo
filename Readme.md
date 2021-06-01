###########################################################################


							Sales Prediction							


###########################################################################

#  Solution :
A Tool to predict Future sales by store and item level.
A Flask application deployed into a docker container that contain a xgboost classification model for sales prediction.
#  How to use:
	The tool is ready to move into production
	1)Run the project and create the docker image using the Dockerfile inside the folder:
	***$ docker build --pull --rm -f "Dockerfile" -t endpointcc:latest "."***
	2)Start the docker container by running this command:
	***$ docker run --rm -d  -p 5000:5000/tcp endpointcc:latest***
	3)Run the prediction using this cmd:
		***$ curl -X POST --header "Content-Type: application/json" 'http://127.0.0.1:5000/predict'***