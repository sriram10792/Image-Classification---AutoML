import label_image


image_path="H:\\Articles to share\\Git codes\\Image Classification using AUTOML\\daisy.jpg"


result=label_image.fetch_prediction(image_path,'flower-classification-215723','ICN7082387764171081847')
print (result)
