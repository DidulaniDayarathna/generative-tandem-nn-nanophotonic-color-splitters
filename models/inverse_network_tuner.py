import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Dense, Reshape, Concatenate,
    Conv2D, UpSampling2D, Dropout,
    Flatten, LeakyReLU
)
from tensorflow.keras.models import Model
import keras_tuner as kt


bce = tf.keras.losses.BinaryCrossentropy()

def inverse_loss(y_true, y_pred):

    error_1 = bce(y_true, y_pred)

    y_true_expand = dimension_expand_mod(y_true)
    y_pred_expand = dimension_expand_mod(y_pred)

    predicted_true = resnet_model_comparison(y_true_expand)
    predicted_spec = resnet_model_comparison(y_pred_expand)

    error_2 = tf.keras.losses.MSE(predicted_true, predicted_spec)

    error = (0.1 * error_1) + (0.9 * error_2)

    return error


def build_model(hp):

    spectrum = 9
    channels = 1

    z_dim = hp.Choice("z_dim", [10, 20, 50])

    inputA = Input(shape=(z_dim,))
    inputB = Input(shape=(spectrum,))
    inputC = Input(shape=(channels,))

    alpha = hp.Choice("alpha", [0.03, 0.05, 0.07])

    dense_units = hp.Choice("dense_units", [256*5*5, 128*5*5])

    x = Dense(dense_units)(inputA)
    x = LeakyReLU(alpha=alpha)(x)
    x = Reshape((5, 5, dense_units // 25))(x)

    y = Dense(dense_units)(inputB)
    y = LeakyReLU(alpha=alpha)(y)
    y = Reshape((5, 5, dense_units // 25))(y)

    p = Dense(dense_units)(inputC)
    p = LeakyReLU(alpha=alpha)(p)
    p = Reshape((5, 5, dense_units // 25))(p)

    combined = Concatenate()([x, y, p])

    filters_1 = hp.Choice("filters_1", [128, 256])
    filters_2 = hp.Choice("filters_2", [64, 128])

    z = UpSampling2D()(combined)
    z = Conv2D(filters_1, kernel_size=3, padding='same')(z)
    z = LeakyReLU(alpha=alpha)(z)

    z = UpSampling2D()(combined)
    z = Conv2D(filters_2, kernel_size=3, padding='same')(z)
    z = LeakyReLU(alpha=alpha)(z)

    z = Conv2D(1, kernel_size=3, padding='same')(z)
    z = LeakyReLU(alpha=alpha)(z)

    dropout_rate = hp.Choice("dropout", [0.5, 0.75])
    z = Dropout(dropout_rate)(z)

    z = Flatten()(z)

    dense_final = hp.Choice("dense_final", [4872, 4096])
    z = Dense(dense_final)(z)

    z = Reshape((56, 29, 3))(z)

    output_activation = hp.Choice("output_activation", ["sigmoid", "linear"])
    z = Conv2D(3, kernel_size=3, padding='same', activation=output_activation)(z)

    model = Model(inputs=[inputA, inputB, inputC], outputs=z)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=inverse_loss
    )

    return model



def run_tuner(noise_train, trainy_norm, train_channel, Trainx):

    tuner = kt.RandomSearch(
        build_model,
        objective="val_loss",
        max_trials=10,
        executions_per_trial=1,
        directory="tuner_results",
        project_name="inverse_network"
    )

    tuner.search(
        [noise_train, trainy_norm, train_channel],
        Trainx,
        validation_split=0.3,
        epochs=60,
        batch_size=32,
        verbose=1
    )

    best_model = tuner.get_best_models(num_models=1)[0]

    return best_model