import kfp
from kfp.v2 import dsl,compiler
from kfp.v2.dsl import component,pipeline,Output,Artifact,Input,Model


CUSTOM_SERVING_CONTAINER  = "us-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.1-0:latest"
@component(base_image            = CUSTOM_SERVING_CONTAINER,
           #output_component_file = 'gs://gcp-practice-0123-18jun2023/docker-kfp-test/yaml/deploy_model.yaml'
           #output_component_file = '/home/jupyter/KFP-json2/yaml/deploy_model.yaml'
           output_component_file = 'E:\\Python-code\\VertexAI-MLPipelines\\yaml\\deploy_model.yaml',
           packages_to_install   = ['google-cloud-aiplatform']
          )
def deploy_model(project_in:str,
                  model_display_name_in : str,
                  model_resource_nm_in:str,
                  endpoint_in : str
                ):
    print("\n\n***Deploying Model....")
    from google.cloud import aiplatform
    aiplatform.init(project=project_in)
    model_obj = aiplatform.Model(model_resource_nm_in)
    endpoin_obj = aiplatform.Endpoint(endpoint_in)
    deployed_model = model_obj.deploy(endpoint                    = endpoin_obj,
                                      deployed_model_display_name = model_display_name_in,
                                      min_replica_count           = 1,
                                      max_replica_count           = 1,
                                      traffic_split               = {"0":100}
                                     )