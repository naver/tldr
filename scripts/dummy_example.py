import argparse
import numpy as np

from tldr import TLDR


parser = argparse.ArgumentParser('Dummy TLDR example')
parser.add_argument('--device', default='cuda', type=str)
args = parser.parse_args()

# Training
X = np.random.rand(100000, 2048)  # replace with training NxD array

tldr = TLDR(
    n_components=32,
    n_neighbors=5,
    encoder="linear",
    projector="mlp-1-2048",
    device=args.device,
    verbose=2,
    knn_approximation="medium",
)
tldr.fit(X, epochs=20, warmup_epochs=5, batch_size=1024, output_folder="data/", print_every=100)
Z = tldr.transform(X, l2_norm=True)  # Returns Nxn_components array

tldr.save("data/inference_model.pth")
tldr.save_knn("data/knn.npy")  # We can save the pre-computed KNN for future trainings with this data

# Inference
X = np.random.rand(5000, 2048)  # replace with test NxD matrix
tldr = TLDR()
tldr.load("data/inference_model.pth", init=True)  # With init=True Loads both model parameters and weights
Z = tldr.transform(X, l2_norm=True)  # Returns a Nxn_components array
