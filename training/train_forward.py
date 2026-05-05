from models.forward_network_tuner import run_tuner
from utils.data_utils import load_forward_data


def main():

    trainxr_final, trainyr_norm = load_forward_data()

    best_model = run_tuner(trainxr_final, trainyr_norm)
    best_model.save("resnet_model_comparison.h5")


if __name__ == "__main__":
    main()