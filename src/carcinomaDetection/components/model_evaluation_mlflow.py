from pathlib import Path
import tensorflow as tf 
import dagshub
import mlflow
import mlflow.keras
from urllib.parse import urlparse
from carcinomaDetection.entity.config_entity import EvaluationConfig
from carcinomaDetection.utils.common import save_json


class Evaluation:
  def __init__(self, config: EvaluationConfig):
    self.config = config

  def _valid_generator(self):

    datagenerator_kwargs = dict(
        rescale = 1./255,
    )

    dataflow_kwargs = dict(
        target_size=self.config.params_image_size[:-1],
        batch_size=self.config.params_batch_size,
        interpolation="bilinear"
    )

    valid_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
        **datagenerator_kwargs
    )

    self.valid_generator = valid_datagenerator.flow_from_directory(
        directory=self.config.testing_data,
        shuffle=False,
        **dataflow_kwargs
    )

  @staticmethod
  def load_model(path: Path) -> tf.keras.Model:
    return tf.keras.models.load_model(path)

  def evaluation(self):
    self.model = self.load_model(self.config.path_of_model)
    self._valid_generator()
    self.score = self.model.evaluate(self.valid_generator)

  def save_score(self):
    scores = {"loss": self.score[0], "accuracy": self.score[1]}
    save_json(path=Path("scores.json"), data=scores)

  def log_into_mlflow(self):
    dagshub.init(repo_owner='quantumsleeper', repo_name='Adenocarcinoma-Detection', mlflow=True)

    with mlflow.start_run():
      mlflow.log_params(self.config.all_params)
      mlflow.log_metrics({"loss": self.score[0], "accuracy": self.score[1]})
      
      mlflow.keras.log_model(self.model, "model", registered_model_name="VGG16")