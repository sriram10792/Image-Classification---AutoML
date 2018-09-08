#Using Google AutoML API for image classification

import io

#Import google cloud AUTOML
from google.cloud import automl_v1beta1
#from google.cloud.automl_v1beta1.proto import service_pb2

#Create a service account and provide the json key corresponding to the project
import os
os.environ["GOOGLE_APPLICATION_CREDENTIALS"]="automl_json.json"

#Providee the image, project id from google cloud, model id from AUTOML
def get_prediction(content, project_id, model_id):
  prediction_client = automl_v1beta1.PredictionServiceClient()

  name = 'projects/{}/locations/us-central1/models/{}'.format(project_id, model_id)
  payload = {'image': {'image_bytes': content }}
  params = {}
  #The response has the image classification and the confidence score for that image
  response = prediction_client.predict(name, payload, params)
  labels = response.payload # Get labels from response
  #Get the classification of that image from labels
  image_class=labels[0].display_name
  #score_result has  has the confidence score for that category
  score_result=labels[0].classification
  confidence=score_result.score
  #print(labels)
  return image_class,confidence  # waits till request is returned

def fetch_prediction(content,project_id,model_id):
    file_path = content
    project_id = project_id
    model_id = model_id

    with open(file_path, 'rb') as ff:
        content = ff.read()

    classification,confidence=get_prediction(content, project_id,  model_id)
    #results={classification:confidence}
    #results=get_prediction(content, project_id,  model_id)
    return classification,confidence
    
