import tensorflow as tf
import numpy as np
import json
import flops_calculator as fc
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, roc_auc_score, precision_score, classification_report

(_, _), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
x_test = x_test / 255
# y_test = tf.keras.utils.to_categorical(y_test, 10)

f = open("/m100/home/userexternal/mfraccar/Symbolic_DNN-Tuner/Model/model-1653057599.541585.json")
mj = json.load(f)
model_json = json.dumps(mj)
model = tf.keras.models.model_from_json(model_json)
model.load_weights(
    "/m100/home/userexternal/mfraccar/Symbolic_DNN-Tuner/Weights/weights-1653057599.541585.h5")

print(model.summary())

flops, _ = fc.analyze_model(model)

print("FLOPS ATTUALI {} - FLOPS MASSIMI {}".format(flops.total_float_ops, 77479996))

model.compile(optimizer="adamax", loss="categorical_crossentropy", metrics=["accuracy"])
row_preds = model.predict(x_test)
# print(preds)
preds = np.argmax(model.predict(x_test), axis=1)

print(row_preds.shape)
print(preds.shape)
print(tf.squeeze(y_test).shape)
print("ACCURACY: ", accuracy_score(y_test, preds))
print("ROC_AUC: ", roc_auc_score(tf.squeeze(y_test), row_preds, multi_class="ovr"))
print("Precision: ", precision_score(y_test, preds))

