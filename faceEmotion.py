from tensorflow.keras.models import model_from_json
import numpy as np


class FacialExpressionModel(object):

    EMOTIONS_LIST = ["Angry", "Disgust",
                     "Fear", "Happy",
                     "Neutral", "Sad",
                     "Surprise"]

    def __init__(self, model_json_file="FER_CONFIG//FER_arch.json", model_weights_file = "FER_CONFIG\FER_weights.h5"):
        # load model from JSON file
        with open(model_json_file, "r") as json_file:
            loaded_model_json = json_file.read()
            self.loaded_model = model_from_json(loaded_model_json)

        # load weights into the new model
        self.loaded_model.load_weights(model_weights_file)

    def predict_emotion(self, img):
        self.preds = self.loaded_model.predict(np.array([img]))
        return FacialExpressionModel.EMOTIONS_LIST[np.argmax(self.preds)]
