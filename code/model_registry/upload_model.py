import kfp
from kfp.v2 import dsl,compiler
from kfp.v2.dsl import component,pipeline,Output,Artifact,Input,Model

CUSTOM_SERVING_CONTAINER  = "us-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.1-0:latest"


@component(base_image            = CUSTOM_SERVING_CONTAINER,
           #output_component_file = 'gs://gcp-practice-0123-18jun2023/docker-kfp-test/yaml/register_model.yaml'
           #output_component_file = '/home/jupyter/KFP-json2/yaml/register_model.yaml'
           output_component_file = 'E:\\Python-code\\VertexAI-MLPipelines\\yaml\\register_model.yaml',
           packages_to_install   = ['google-cloud-aiplatform']
          )
def register_model(project_in:str,
                    model_display_name_in : str,
                    model_gcs_path_in : str,
                    serving_container_in : str
                  ) -> str:
    print("\n\n***Registering Model....")
    from google.cloud import aiplatform
    aiplatform.init(project=project_in)
    my_model = aiplatform.Model.upload( project                     = project_in,
                                        display_name                = model_display_name_in,
                                        serving_container_image_uri = serving_container_in,
                                        artifact_uri                = model_gcs_path_in
                                      )
    print("***Model Resource Name is : ", my_model.resource_name)
    return my_model.resource_name