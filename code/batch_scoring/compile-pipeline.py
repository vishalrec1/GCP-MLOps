import kfp
from kfp.v2 import dsl,compiler
from kfp.v2.dsl import component,pipeline,Output,Artifact,Input,Model

from make_pipeline import mlopsDeployPipeline

compiler.Compiler().compile(pipeline_func = mlopsDeployPipeline,
							#package_path  = 'gs://gcp-practice-0123-18jun2023/docker-kfp-test/kfp-json/mlopsdeploypipeline-1.json'
                            package_path  = "E:\\Python-code\\VertexAI-MLPipelines\\json\\mlopsdeploypipeline-1.json"
						   )