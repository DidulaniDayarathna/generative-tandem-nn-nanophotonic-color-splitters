from models.inverse_network_tuner import run_tuner
from utils.data_utils import load_data


def main():

    noise_train, trainy_norm, train_channel, Trainx = load_data()

    best_model = run_tuner(
        noise_train,
        trainy_norm,
        train_channel,
        Trainx
    )

    best_model.save("best_inverse_model.h5")


if __name__ == "__main__":
    main()