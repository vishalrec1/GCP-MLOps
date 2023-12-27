from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import pandas as pd
import pickle
from google.cloud import storage
import os

df=pd.read_csv('/gcs/nonprod-corp-gcs-1cdh-234446-01-aimldev/customtrain_test/california_housing_train.csv')
labels = df.pop('median_house_value').tolist()
data = df.values.tolist()
x_train, x_test, y_train, y_test = train_test_split(data, labels)
skmodel = LinearRegression()
skmodel.fit(x_train,y_train)
score = skmodel.score(x_test,y_test)
print('accuracy is:',score)

artifact_filename = 'model.pkl'
print(artifact_filename)

# Save model artifact to local filesystem (doesn't persist)
local_path = artifact_filename
with open(local_path, 'wb') as model_file:
  pickle.dump(skmodel, model_file)

# Upload model artifact to Cloud Storage
model_directory = os.environ['AIP_MODEL_DIR']
print(model_directory)
storage_path = os.path.join(model_directory, artifact_filename)
print(storage_path)
blob = storage.blob.Blob.from_string(storage_path, client=storage.Client())
blob.upload_from_filename(local_path)