import mlflow
from bertopic import BERTopic

class Bert_model(mlflow.pyfunc.PythonModel):

    def load_context(self, context):
       # print(context)

          self.model = BERTopic.load("/content/drive/MyDrive/model.bertopic")
    def predict(self, context, model_input):
        return self.model.transform(model_input)

    def visualize(self):
        return self.model.visualize_topics()
