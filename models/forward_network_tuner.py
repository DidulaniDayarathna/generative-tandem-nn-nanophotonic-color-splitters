import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.models import Model
import keras_tuner as kt


def build_model(hp):

    image_size = [112, 58]

    resnet = ResNet50(
        input_shape=image_size + [3],
        weights='imagenet',
        include_top=False
    )

    for layer in resnet.layers:
        layer.trainable = False

    dense_1_units = hp.Choice("dense_1_units", [704, 512])
    dense_2_units = hp.Choice("dense_2_units", [384, 256])

    dropout_1 = hp.Choice("dropout_1", [0.15, 0.2])
    dropout_2 = hp.Choice("dropout_2", [0.15, 0.2])

    x = Flatten()(resnet.output)

    x = Dense(dense_1_units, activation='tanh')(x)
    x = Dropout(dropout_1)(x)

    x = Dense(dense_2_units, activation='relu')(x)
    x = Dropout(dropout_2)(x)

    prediction = Dense(9, activation='linear')(x)

    model = Model(inputs=resnet.input, outputs=prediction)

    learning_rate = hp.Float("learning_rate", 1e-4, 1e-1, sampling="log")

    decay_rate = learning_rate / 270  

    optimizer = tf.keras.optimizers.Adam(
        learning_rate=decay_rate,
        beta_1=0.5
    )

    model.compile(
        loss='mean_squared_error',
        optimizer=optimizer
    )

    return model

def run_tuner(trainxr_final, trainyr_norm):

    tuner = kt.RandomSearch(
        build_model,
        objective="val_loss",
        max_trials=10,
        executions_per_trial=1,
        directory="tuner_results",
        project_name="forward_network"
    )

    tuner.search(
        x=trainxr_final,
        y=trainyr_norm,
        batch_size=32,
        epochs=500,
        validation_split=0.2,
        verbose=1
    )

    best_model = tuner.get_best_models(num_models=1)[0]

    return best_model