from google.cloud import aiplatform

PROJECT_ID                 = 'gcp-practice-0123'
PREBUILT_SERVING_CONTAINER = "us-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.1-0:latest"
pipeline_root_path         = 'gs://gcp-practice-0123-18jun2023/test-kfp/kfp/'

mlops_pipeline_job = aiplatform.PipelineJob(display_name    = "model-deployment-pipeline-test-json",
                                            pipeline_root   = pipeline_root_path,
                                            #template_path   = "gs://gcp-practice-0123-18jun2023/docker-kfp-test/kfp-json/mlopsdeploypipeline-1.json",
                                            template_path   =  "E:\\Python-code\\vertex ai notes\\json\\mlopsdeploypipeline-1.json"
                                            project         = PROJECT_ID,
                                            parameter_values={"project_in":PROJECT_ID,
                                                              "endpoint_display_name_in":"mlopsDeployPipeline-1",
                                                              "model_gcs_path_in":'gs://gcp-practice-0123-18jun2023/custom-trained-model/model/aiplatform-custom-training-2023-01-22-18:07:17.227/model/',
                                                              "serving_container_in":PREBUILT_SERVING_CONTAINER,
                                                              "bigquery_source_in":"bq://gcp-practice-0123.dataset2.california_housing_test",
                                                              "bigquery_destination_prefix_in":'bq://gcp-practice-0123.dataset2',
                                                              "predictions_format_in":"bigquery",
                                                              "machine_type_in":"e2-standard-4",
                                                              "bucket_uri_in":"gs://gcp-practice-0123-18jun2023/custom-trained-model/training-data"
                                                             }
                                            
                                          )

mlops_pipeline_job.submit(service_account='sa-nonprod-corp-1cdh-214e-01@nonprod-corp-1cdh-214e.iam.gserviceaccount.com')