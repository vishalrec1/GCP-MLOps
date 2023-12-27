from google.cloud import aiplatform

PROJECT_ID                 = 'nonprod-corp-1cdh-214e'
REGION					   = 'us-central1'
BUCKET_URI

aiplatform.init(project=PROJECT_ID, location=REGION,staging_bucket=BUCKET_URI)
custom_model_job = aiplatform.CustomTrainingJob(display_name='Custom_Trained_Model',
                                                script_path='/home/jupyter/custom_training_model.py',
                                                container_uri="us-central1-docker.pkg.dev/nonprod-corp-1cdh-214e/containers/1cdh_mlops_image:r6",
                                                requirements=["gcsfs",],
                                                model_serving_container_image_uri="us-central1-docker.pkg.dev/nonprod-corp-1cdh-214e/containers/1cdh_mlops_image:r6")
												
custom_model = custom_model_job.run(model_display_name='Custom_Trained_Model_Test1',
                                    replica_count=10,
                                    machine_type=MACHINE_TYPE,
                                    sync=False)
									
custom_model_endpoint = custom_model.deploy(machine_type=MACHINE_TYPE,sync=False)