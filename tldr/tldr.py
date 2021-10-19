# Copyright (c) NAVER and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import json
import random
from pathlib import Path
from time import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import BatchSampler, RandomSampler

from tldr.loss import BarlowTwinsLoss
from tldr.optimizer import LARS, adjust_learning_rate
from tldr.utils import AverageMeter, get_knn_graph, get_progress_bar, parse_net_config


class TLDR_Module(nn.Module):
    def __init__(
        self,
        inputdim,
        n_components,
        encoder="linear",
        projector="mlp-2-2048",
        batch_size=1024,
        scale_loss=1.0 / 32,
        lambd=3.9e-3,
        norm_layer="BN",
        loss="BT",
    ):
        super().__init__()

        self.batch_size = batch_size
        self.scale_loss = scale_loss
        self.lambd = lambd
        self.loss = loss

        if norm_layer == "BN":
            self.norm_layer = nn.BatchNorm1d
        elif norm_layer == "LN":
            self.norm_layer = nn.LayerNorm

        # Encoder
        encoder_type, num_hlayers_encoder, hdims_encoder = parse_net_config(encoder)
        hdims = hdims_encoder * num_hlayers_encoder
        hdims = [inputdim] + hdims + [n_components]
        layers = []
        if encoder_type in ["linear", "flinear"]:
            for i in range(len(hdims) - 2):
                layers.append(nn.Linear(hdims[i], hdims[i + 1]))
                layers.append(self.norm_layer(hdims[i + 1]))
            layers.append(nn.Linear(hdims[-2], hdims[-1]))
        elif encoder_type == "mlp":
            for i in range(len(hdims) - 2):
                layers.append(nn.Linear(hdims[i], hdims[i + 1], bias=False))
                layers.append(self.norm_layer(hdims[i + 1]))
                layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Linear(hdims[-2], hdims[-1], bias=False))
        else:
            raise ValueError(f"Incorrect network type {encoder_type}")
        self.encoder = nn.Sequential(*layers)

        # Projector
        if projector is not None:
            projector_type, num_hlayers_projector, hdims_projector = parse_net_config(projector)
            sizes = [n_components] + hdims_projector * (num_hlayers_projector + 1)
            layers = []
            if projector_type in ["linear", "flinear"]:
                for i in range(len(sizes) - 2):
                    layers.append(nn.Linear(sizes[i], sizes[i + 1]))
                    layers.append(self.norm_layer(sizes[i + 1]))
                layers.append(nn.Linear(sizes[-2], sizes[-1]))
            elif projector_type == "mlp":
                for i in range(len(sizes) - 2):
                    layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=False))
                    layers.append(self.norm_layer(sizes[i + 1]))
                    layers.append(nn.ReLU(inplace=True))
                layers.append(nn.Linear(sizes[-2], sizes[-1], bias=False))
            else:
                raise ValueError(f"Incorrect network type {projector_type}")
            self.projector = nn.Sequential(*layers)
            bn_size = sizes[-1]
        else:
            bn_size = n_components
            self.projector = None

        # normalization layer for the representations z1 and z2
        self.bn = nn.BatchNorm1d(bn_size, affine=False)

    def forward(self, X):
        return self.encoder(X)

    def match(self, y1, y2):
        z1 = self.encoder(y1)
        z2 = self.encoder(y2)
        if self.projector is not None:
            z1 = self.projector(z1)
            z2 = self.projector(z2)

        if self.loss in ["BT", "BarlowTwins"]:
            loss = BarlowTwinsLoss(self.bn(z1), self.bn(z2), self.batch_size, self.scale_loss, self.lambd)
        elif self.loss in ["MSE", "MeanSquaredError"]:
            loss = nn.MSELoss(reduction="mean")(torch.vstack([y1, y2]), torch.vstack([z1, z2])).mul(self.scale_loss)
        elif self.loss == "Contrastive":
            raise ValueError("Contrastive loss temporary removed :_( (WIP)")
        return loss.unsqueeze(0)

    def set_device(self, device):
        self.encoder.to(device)
        self.projector.to(device)


class TLDR:
    def __init__(
        self,
        n_components=32,
        encoder="linear",
        projector="mlp-2-2048",
        n_neighbors=5,
        device="cpu",
        pin_memory=False,
        knn_graph=None,
        inputdim=None,
        batch_size=1024,
        scale_loss=1.0 / 32,
        lambd=3.9e-3,
        epochs=100,
        learning_rate=0.2,
        warmup_epochs=10,
        norm_layer="BN",
        loss="BT",
        gaussian=False,
        output_folder=None,
        snapshot_freq=0,
        resume=False,
        writer=None,
        save_best=False,
        verbose=0,
        random_state=None,
        knn_approximation=None,
    ):
        self.architecture = {
            "inputdim": inputdim,
            "n_components": n_components,
            "encoder": encoder,
            "projector": projector,
            "batch_size": batch_size,
            "scale_loss": scale_loss,
            "lambd": lambd,
            "norm_layer": norm_layer,
            "loss": loss,
        }
        self.model = None
        self.device = torch.device(device) if type(device) == str else device
        self.batch_size = batch_size
        self.epochs = epochs
        self.start_epoch = 0
        self.n_neighbors = n_neighbors
        self.learning_rate = learning_rate
        self.knn_graph = knn_graph
        self.warmup_epochs = warmup_epochs
        self.snapshot_freq = snapshot_freq
        self.output_folder = output_folder
        self.resume = resume
        self.pin_memory = pin_memory
        self.gaussian = gaussian
        self.writer = writer
        self.save_best = save_best
        self.verbose = verbose
        self.random_state = random_state
        self.knn_approximation = knn_approximation
        if knn_approximation not in [None, "low", "medium", "high"]:
            raise ValueError(
                f"knn_approximation should be either None or low, medium, or high and it was {knn_approximation}"
            )
        if self.random_state is not None:
            torch.manual_seed(self.random_state)
            np.random.seed(self.random_state)
            random.seed(self.random_state)

    def initialize_model(self):
        self.model = TLDR_Module(**self.architecture)
        self.model.to(self.device)

    def parameters(self):
        if self.model is None:
            raise RuntimeError("model not initialized")

        def concat_generators(*args):
            for gen in args:
                yield from gen

        if self.model.projector is not None:
            return concat_generators(self.model.encoder.parameters(), self.model.projector.parameters())
        else:
            return self.model.encoder.parameters()

    def fit(
        self,
        X,
        epochs=None,
        warmup_epochs=None,
        batch_size=None,
        knn_graph=None,
        output_folder=None,
        snapshot_freq=None,
        dataset_val=None,
        l2_norm_eval=False,
        print_every=None,
        eval_every=None,
        retain_projector=False,
    ):
        self.architecture["inputdim"] = X.shape[1]
        if epochs is not None:
            self.epochs = epochs
        if warmup_epochs is not None:
            self.warmup_epochs = warmup_epochs
        if batch_size is not None:
            self.batch_size = batch_size
            self.architecture["batch_size"] = batch_size
        if output_folder is not None:
            self.output_folder = Path(output_folder)
        if self.output_folder is not None:
            self.output_folder.mkdir(parents=True, exist_ok=True)
        if snapshot_freq is not None:
            self.snapshot_freq = snapshot_freq

        self.initialize_model()
        if self.model is None:
            raise RuntimeError("model not initialized")

        if self.verbose > 1:
            if "cuda" in self.device.type:
                print(" - Using GPU")
            else:
                print(" - Using CPU")
        n_data = X.shape[0]

        if knn_graph is not None:
            self.knn_graph = knn_graph
        elif self.knn_graph is None:
            self.compute_knn(X)

        # Resuming options
        if self.resume:
            path = self.output_folder / "final_model.pth"
            if path.is_file():
                self.load(path)
                print(" * Final model found. Skipping training.")
                return
            path = self.output_folder / "latest_snapshot.pth"
            if path.is_file():
                self.load(path)
        self.model.train()

        if isinstance(X, np.ndarray):  # if data is not a tensor convert it
            X = torch.Tensor(X)
        X = X.float()
        if self.pin_memory:
            X = X.to(self.device)

        def exclude_bias_and_norm(p):
            return p.ndim == 1

        optimizer = LARS(
            self.model.parameters(),
            lr=0,
            weight_decay=1e-6,
            weight_decay_filter=exclude_bias_and_norm,
            lars_adaptation_filter=exclude_bias_and_norm,
        )

        losses = AverageMeter("Loss", ":.4e")
        batch_sampler = BatchSampler(RandomSampler(range(n_data)), batch_size=self.batch_size, drop_last=True)
        step = self.start_epoch * len(batch_sampler)
        best_eval = 0
        t0 = time()
        with get_progress_bar() as progress:
            task = (
                progress.add_task(
                    description="[green]Training TLDR", total=(len(batch_sampler) * self.epochs), info="-"
                )
                if self.verbose > 0
                else None
            )
            for epoch in range(self.start_epoch, self.epochs):
                if self.verbose > 0:
                    progress.update(task, info=f"epoch {epoch+1} (of {self.epochs})")
                for i, ind in enumerate(batch_sampler):
                    step += 1
                    if type(self.knn_graph) == dict:  # Oracle
                        ind_nn = []
                        for j in ind:
                            ind_nn.append(random.choices(self.knn_graph[j])[0])
                        y1 = X[ind, :]
                        y2 = X[ind_nn, :]
                    else:
                        if self.gaussian:  # Synthetic neighbours
                            y1 = X[ind, :]
                            y2 = y1 + (torch.std(y1) ** 0.5) * torch.randn(y1.shape).to(self.device) * 0.1
                        else:  # Randomly select m neighbours as training pair(s)
                            y1 = X[ind, :]
                            ind_nn = np.random.randint(self.n_neighbors, size=self.batch_size)
                            y2 = X[self.knn_graph[ind, ind_nn], :]

                    if not self.pin_memory:
                        y1 = y1.to(self.device)
                        y2 = y2.to(self.device)

                    lr = adjust_learning_rate(
                        self.epochs,
                        optimizer,
                        n_data,
                        step,
                        self.learning_rate,
                        self.batch_size,
                        self.warmup_epochs,
                    )
                    optimizer.zero_grad()
                    loss = self.model.match(y1, y2).mean()
                    losses.update(loss.item(), y1.size(0))
                    loss.mean().backward()
                    optimizer.step()
                    if print_every and step % print_every == 0:
                        if self.verbose > 1:
                            progress.console.print(f" * {losses}, LR = {lr:.5f}")
                        if self.writer:
                            self.writer.add_scalar(
                                f'n{self.architecture["n_components"]}/train/loss',
                                losses.val,
                                epoch + (i / len(batch_sampler)),
                            )
                    if self.verbose > 0:
                        progress.update(task, advance=1)
                checkpoint = {
                    "epoch": epoch + 1,
                    "state_dict": self._get_state_dict(),
                    "architecture": self.architecture,
                }

                if dataset_val is not None and (epoch + 1) % eval_every == 0:
                    res = self.evaluate(dataset_val, l2_norm_eval)
                    checkpoint["val"] = res
                    if self.writer:
                        self.writer.add_scalar(
                            f'n{self.architecture["n_components"]}/val/acc',
                            res,
                            epoch + 1,
                        )
                    if res > best_eval and self.output_folder and self.save_best:
                        torch.save(checkpoint, self.output_folder / "best.pth")
                        best_eval = res
                    self.model.train()

                if self.output_folder:
                    if self.snapshot_freq and (epoch + 1) % self.snapshot_freq == 0:
                        torch.save(checkpoint, self.output_folder / f"snapshot_{epoch+1}.pth")
                    torch.save(checkpoint, self.output_folder / "latest_snapshot.pth")

            if self.output_folder:
                torch.save(checkpoint, self.output_folder / "final_model.pth")
        t1 = time()
        if self.verbose > 1:
            print(" - Fit took %.2g sec" % (t1 - t0))
        if not retain_projector:
            self.remove_projector()

    def transform(self, X, l2_norm=False, batching_threshold=10000, amount_batches=1000):
        if self.model is None:
            raise RuntimeError("model not initialized")

        with torch.no_grad():  # Avoid computing gradients when we do not need them
            self.model.eval()
            self.model.to(self.device)
            if isinstance(X, np.ndarray):  # If data is not a tensor convert it
                X = torch.Tensor(X)
                to_numpy = True  # Output type same as input
                input_device = "cpu"
            elif isinstance(X, torch.Tensor):
                to_numpy = False  # Output type same as input
                input_device = X.device.type
            else:
                raise ValueError(f"unknow input type {type(X)}. Input must be numpy array or torch tensor")

            if (
                X.shape[0] > batching_threshold
            ):  # If there are more than batching_threshold samples do batched transformation
                splitted_dataset = torch.split(X, amount_batches)
                all_data = list()
                for batch in splitted_dataset:
                    all_data.append(self.forward(batch, l2_norm=l2_norm))
                Z = torch.vstack(all_data)
            else:
                Z = self.forward(X, l2_norm=l2_norm)
        if input_device == "cpu":
            Z = Z.cpu()
        if to_numpy:
            Z = Z.detach().numpy()
        return Z

    def fit_transform(self, X, **kwargs):
        """
        See documentation of methods fit() and transform()
        """
        l2_norm = kwargs.pop("l2_norm", False)
        batching_threshold = kwargs.pop("batching_threshold", 10000)
        amount_batches = kwargs.pop("amount_batches", 1000)
        self.fit(X, **kwargs)
        return self.transform(X, l2_norm=l2_norm, batching_threshold=batching_threshold, amount_batches=amount_batches)

    def forward(self, X, l2_norm=False):
        Z = self.model.forward(X.float().to(self.device))
        if l2_norm:
            Z = Z / torch.linalg.norm(Z, 2, axis=1, keepdims=True)
        return Z

    def save(self, path):
        path = Path(path)
        if not path.parent.is_dir():
            path.parent.mkdir(parents=True, exist_ok=True)

        architecture = self.architecture
        if self.model.projector is None:
            architecture["projector"] = None
        checkpoint = {
            "state_dict": self._get_state_dict(),
            "architecture": architecture,
        }
        torch.save(checkpoint, path)

    def load(self, path, init=False, strict=False):
        checkpoint = torch.load(path, map_location=torch.device("cpu"))
        architecture = checkpoint.pop("architecture", None)
        if init:
            self.architecture = architecture
            self.initialize_model()
        else:
            if architecture != self.architecture:
                raise ValueError(
                    f"Parameters in {path} do not match. Use load(path, init=True) to intialize the model with the parameters in this file."
                )

        self.model.load_state_dict(checkpoint["state_dict"], strict=strict)
        if "epoch" in checkpoint:
            self.start_epoch = checkpoint["epoch"]
        self.model.to(self.device)

    def compute_knn(self, X):
        self.knn_graph = get_knn_graph(
            X, self.n_neighbors, device=self.device.type, verbose=self.verbose, knn_approximation=self.knn_approximation
        )

    def get_knn(self):
        return self.knn_graph

    def save_knn(self, path):
        path = Path(path)
        if not path.parent.is_dir():
            path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, self.knn_graph)

    def load_knn(self, path):
        self.knn_graph = np.load(path)

    def _get_state_dict(self):
        return self.model.state_dict()

    def to(self, device):
        self.device = torch.device(device)
        self.model.to(self.device)

    def remove_projector(self):
        self.model.projector = None
        self.model.bn = None

    def evaluate(self, dataset, l2_norm_eval=True, whiten_eval=False, metric="mAP-medium"):
        self.model.eval()
        dataset.transform(self)
        return dataset.evaluate(l2_norm=l2_norm_eval, whiten=whiten_eval, metric=metric)

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        s = f"{self.__class__.__name__}\n"
        s += f"{json.dumps(self.architecture, indent=2)}\n\n{self.model}"
        return s
