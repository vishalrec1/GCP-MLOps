import kfp
from kfp.v2 import dsl,compiler
from kfp.v2.dsl import component,pipeline,Output,Artifact,Input,Model

#pipeline_root_path          = 'gs://gcp-practice-0123-18jun2023/docker-kfp-test/kfp-json/'
pipeline_root_path          = "E:\\Python-code\\VertexAI-MLPipelines\\json\\"
@dsl.pipeline(name          = "mlopsdeploypipeline-1",
              pipeline_root = pipeline_root_path
             )
def mlopsDeployPipeline(project_in:str,
                        endpoint_display_name_in:str,
                        model_gcs_path_in:str,
                        serving_container_in:str,
                        bigquery_source_in:str,
                        bigquery_destination_prefix_in:str,
                        predictions_format_in:str,
                        machine_type_in:str,
                        bucket_uri_in:str
                       ):
    from create_endpoint import create_endpoint
    from upload_model import register_model
    from deploy_model import deploy_model
    from batch_prediction import batch_prediction
    from loadBigQueryTableFromGCS import loadBigQueryTableFromGCS
    
    endpoint_op = create_endpoint(endpoint_display_name_in,project_in)
    
    model_op = register_model(project_in,endpoint_display_name_in,model_gcs_path_in,serving_container_in)

    deploy_model(project_in,endpoint_display_name_in,model_op.output,endpoint_op.output)
    
    loadBigQueryTableFromGCS(project_in,bucket_uri_in)
    
    batch_prediction(project_in,endpoint_display_name_in,model_op.output,bigquery_source_in,bigquery_destination_prefix_in,predictions_format_in,machine_type_in)