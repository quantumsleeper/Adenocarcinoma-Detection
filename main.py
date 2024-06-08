from carcinomaDetection import logger 
from carcinomaDetection.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
from carcinomaDetection.pipeline.stage_02_prepare_base_model import PrepareBaseModelTrainingPipeline
from carcinomaDetection.pipeline.stage_03_model_trainer import ModelTrainingPipeline
from carcinomaDetection.pipeline.stage_04_model_evaluation import EvaluationPipeline

STAGE_NAME = "Data Ingestion"

try:
    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
    obj = DataIngestionTrainingPipeline()
    obj.main()
    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
    logger.exception(e)
    raise e


STAGE_NAME = "Prepare Base Model"

try:
    logger.info(f">>>>>> Stage {STAGE_NAME} started <<<<<<")
    obj = PrepareBaseModelTrainingPipeline()
    obj.main()
    logger.info(f">>>>>> Stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
    logger.exception(e)
    raise e


STAGE_NAME = "Model Training"


try:
    logger.info(f">>>>>> Stage {STAGE_NAME} started <<<<<<")
    obj = ModelTrainingPipeline()
    obj.main()
    logger.info(f">>>>>> Stage {STAGE_NAME} completed <<<<<<\n\nx==========x")

except Exception as e:
    logger.exception(e)
    raise e


STAGE_NAME = "Model Evaluation"

try:
    logger.info(f"*****************")
    logger.info(f">>>>> Stage {STAGE_NAME} started <<<<<")
    obj = EvaluationPipeline()
    obj.main()
    logger.info(f">>>>> Stage {STAGE_NAME} completed!<<<<<\n\nx==========x")
except Exception as e:
    logger.exception(e)
    raise e
