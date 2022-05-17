import torch
from torch_geometric.data import Batch
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GCN2Conv, GINConv, SAGEConv
import pytorch_lightning as pl
from torchmetrics import Accuracy, F1
import gin


@gin.configurable
class GraphConvolution(pl.LightningModule):
    @gin.configurable(denylist=["num_features", "hidden_channels", "num_classes"])
    def __init__(
        self,
        num_classes,
        num_features,
        hidden_channels,
        conv_layer="GCN2Conv",
        num_layers=64,
        dropout=0.0,
        learning_rate=0.001,
    ):
        super(GraphConvolution, self).__init__()
        self.save_hyperparameters()
        # Initialize the layers
        self.lins = torch.nn.ModuleList()
        self.lins.append(Linear(num_features, hidden_channels))
        self.lins.append(Linear(hidden_channels, num_classes))
        self.convs = torch.nn.ModuleList()
        for layer in range(num_layers):
            if conv_layer == "GCNConv":
                l = GCNConv(in_channels=hidden_channels, out_channels=hidden_channels)
            elif conv_layer == "GCN2Conv":
                alpha = 0.1
                theta = 0.4
                l = GCN2Conv(
                    hidden_channels,
                    alpha,
                    theta,
                    layer + 1,
                    aggr="max",
                )
            elif "GINConv":
                l = GINConv(
                    torch.nn.Sequential(
                        torch.nn.Linear(hidden_channels, hidden_channels),
                        torch.nn.ReLU(),
                        torch.nn.Linear(hidden_channels, hidden_channels),
                    ),
                    eps=0.0,
                    train_eps=True,
                )
            elif conv_layer == "SAGEConv":
                l = SAGEConv(in_channels=hidden_channels, out_channels=hidden_channels)
            else:
                raise ValueError(f"Unknown convolution layer {conv_layer}")
            self.convs.append(l)
        self.train_metric = Accuracy(num_classes=1, multiclass=False)
        self.val_metric = Accuracy(num_classes=1, multiclass=False)
        self.test_metric = Accuracy(num_classes=1, multiclass=False)

    def preprocess_y(self, preds, batch, compute_preds_argmax=True):
        offset = 0
        failures_idx = []
        for i in range(batch.num_graphs):
            data = batch.get_example(i)
            vars = data.failure_modes_idx.values()
            failures_idx.extend([offset + var_idx for var_idx in vars])
            offset += len(data.y)
        y = preds[failures_idx, :]
        if compute_preds_argmax:
            y = torch.argmax(y, dim=1)
        gt = batch.y[failures_idx]
        return y, gt

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = F.dropout(x, self.hparams.dropout, training=self.training)
        x = x_0 = self.lins[0](x).relu()
        edge_weight = torch.ones(
            (edge_index.size(1),), dtype=torch.float, device=edge_index.device
        )

        for conv in self.convs:
            x = F.dropout(x, self.hparams.dropout, training=self.training)
            if self.hparams["conv_layer"] == "GCNConv":
                x = conv(x, edge_index, edge_weight)
            elif self.hparams["conv_layer"] == "GCN2Conv":
                x = conv(x, x_0, edge_index, edge_weight)
            elif  self.hparams["conv_layer"] == "GINConv":
                x = conv(x, edge_index)
            elif  self.hparams["conv_layer"] == "SAGEConv":
                x = conv(x, edge_index)
            x = x.relu()

        x = F.dropout(x, self.hparams.dropout, training=self.training)
        x = self.lins[1](x)

        return x.log_softmax(dim=-1)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)

    def training_step(self, batch: Batch, batch_idx: int):
        y_hat = self(batch)
        loss = F.nll_loss(y_hat, batch.y)
        y_failure_modes, gt_failure_modes = self.preprocess_y(y_hat, batch)
        acc = self.train_metric(y_failure_modes, gt_failure_modes)
        self.log(
            "train_metric",
            self.train_metric,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
        )
        if self.global_step % 100 == 0:
            self.logger.log_metrics(
                metrics={
                    "train_metric": acc.detach().cpu().item(),
                    "loss": loss.detach().cpu().item(),
                },
            )
        return loss

    def validation_step(self, batch: Batch, batch_idx: int):
        y_hat = self(batch)
        y_failure_modes, gt_failure_modes = self.preprocess_y(y_hat, batch)
        acc = self.val_metric(y_failure_modes, gt_failure_modes)
        self.log(
            "val_metric", self.val_metric, prog_bar=True, on_step=False, on_epoch=True
        )
        self.logger.log_metrics(
            metrics={"val_metric": acc.detach().cpu().item()},
        )

    def test_step(self, batch: Batch, batch_idx: int):
        y_hat = self(batch)
        y_failure_modes, gt_failure_modes = self.preprocess_y(y_hat, batch)
        acc = self.test_metric(y_failure_modes, gt_failure_modes)
        self.log(
            "test_metric", self.test_metric, prog_bar=True, on_step=False, on_epoch=True
        )
        self.logger.log_metrics(
            metrics={"test_metric": acc.detach().cpu().item()},
        )
