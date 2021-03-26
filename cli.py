import torch
import typer
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, random_split
from tqdm.auto import trange  # type: ignore
from tqdm.contrib import tenumerate  # type: ignore

from handwriting import data, train
from handwriting.adapters import DataRepository
from handwriting.data import offsets_to_batch
from handwriting.estimator import LSTMGraves, load_estimator, save_estimator
from handwriting.viz import generate_plots

app = typer.Typer(name="handwriting")


@app.command()
def prepare_data() -> None:
    typer.secho("Converting raw data to numpy data", fg="cyan")
    repo = DataRepository()
    typer.secho("-> reading raw files", fg="yellow")
    strokesets = repo.get_all_examples()
    typer.secho("-> converting to numpy data", fg="yellow")
    coords = data.strokesets_to_coords(strokesets)
    coords_aligned = data.align_coords(coords)
    offsets = data.coords_to_offsets(coords_aligned)
    typer.secho("-> saving data", fg="yellow")
    repo.save_numpy_data(offsets)
    typer.secho("-> success!", fg="green")


@app.command()
def train_generator(
    # data
    cv_split: float = 0.9,
    sequence_length: int = 300,
    data_augmentation: bool = True,
    # estimator
    lstm_hidden_size: int = 400,
    lstm_num_layers: int = 3,
    n_mixtures: int = 20,
    # training
    n_epochs: int = 20,
    batch_size: int = 64,
    shuffle: bool = True,
    lr: float = 0.001,
    # lr schedule
    lr_schedule_step_size: int = 100,
    lr_schedule_gamma: float = 0.1,
    # generate
    generate_seq_length: int = 500,
    generate_n_seq: int = 5,
    generate_bias: float = 0.1,
    generate_rainbow: bool = True,
    # others
    device_str: str = "cuda:0" if torch.cuda.is_available() else "cpu",
    normalize_filename: str = "train_normalize",
) -> None:
    typer.secho("Training for unconditional handwriting generation", fg="cyan")
    typer.secho("-> loading data", fg="yellow")
    # init
    device = torch.device(device_str)
    # get data
    repo = DataRepository()
    np_offsets = repo.get_numpy_data()
    np_offsets = data.normalize(np_offsets, filename=normalize_filename)
    # create data split
    len_dataset = np_offsets.x.shape[0]
    train_size = int(cv_split * len_dataset)
    val_size = len_dataset - train_size
    dataset = data.CustomDataset(
        np_offsets, seq_len=sequence_length, data_augmentation=data_augmentation
    )
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_dataloader, val_dataloader = (
        DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle),
        DataLoader(val_dataset, batch_size=batch_size, shuffle=shuffle),
    )
    # estimator
    est = LSTMGraves(
        hidden_size=lstm_hidden_size, n_mdn=n_mixtures, num_layers=lstm_num_layers
    )
    est.to(device)
    optimizer = Adam(est.parameters(), lr=lr)
    lr_scheduler = StepLR(
        optimizer, step_size=lr_schedule_step_size, gamma=lr_schedule_gamma
    )
    typer.secho("-> starting training", fg="yellow")
    batch_idx = 0
    min_val_loss = float("inf")
    tr_epochs = trange(n_epochs, desc="epochs")
    for i_epoch in tr_epochs:
        # epoch training
        est.train()
        batch_losses_sum = 0.0
        tr_batches = trange(len(train_dataloader), desc="batches", leave=False)
        iter_batches = iter(train_dataloader)
        for _ in tr_batches:
            # batch training
            array_data_batch = next(iter_batches)
            batch_data = data.offsets_to_batch(array_data_batch, device=device)
            loss_batch_sum = train.train_one_batch(
                batch_data, est, optimizer, device=device
            )
            loss_batch_sum_float = loss_batch_sum.detach().cpu().item()
            batch_losses_sum += loss_batch_sum_float
            batch_idx += 1
            # updates
            tr_batches.set_description(
                "batches / train {:.2f}".format(
                    loss_batch_sum_float / len(batch_data.offsets)
                )
            )
            tr_batches.refresh()

        train_loss_epoch = batch_losses_sum / train_size
        # calculate validation loss
        batch_losses_sum = 0.0
        est.eval()  # crucial
        with torch.no_grad():
            for i, array_data_batch in tenumerate(
                val_dataloader, desc="loss batches", leave=False
            ):
                batch_data = data.offsets_to_batch(array_data_batch, device=device)
                loss_batch_sum = train.calculate_batch_loss(
                    batch_data, est, train=False, device=device
                )
                batch_losses_sum += loss_batch_sum.detach().cpu().item()
        val_loss_epoch = batch_losses_sum / val_size
        # updates
        lr_scheduler.step()
        if val_loss_epoch < min_val_loss:
            save_estimator(est, filename=f"lstm_epoch_{i_epoch+1:03}")
        min_val_loss = min(val_loss_epoch, min_val_loss)
        generate_plots(
            est,
            sequence_length=generate_seq_length,
            n_sequences=generate_n_seq,
            bias=generate_bias,
            rainbow=generate_rainbow,
            normalize_filename=normalize_filename,
            device_str=device_str,
            base_plot_name=f"epoch_{i_epoch + 1:02}",
        )
        tr_epochs.set_description(
            "epochs / avg train {:.2f} / val {:.2f}".format(
                train_loss_epoch, val_loss_epoch
            )
        )
        tr_epochs.refresh()
    typer.secho("-> success!", fg="green")


@app.command()
def generate(
    # estimator params
    lstm_hidden_size: int = 400,
    lstm_num_layers: int = 3,
    n_mixtures: int = 20,
    estimator_filename: str = "estimator",
    normalize_filename: str = "train_normalize",
    # generation params
    seq_length: int = 500,
    n_seq: int = 5,
    bias: float = 0.1,
    rainbow: bool = True,
    linewidth: int = 3,
    base_filename: str = "generated",
    # others
    device_str: str = "cuda:0" if torch.cuda.is_available() else "cpu",
) -> None:
    typer.secho("Training for unconditional handwriting generation", fg="cyan")
    typer.secho("-> loading estimator", fg="yellow")
    device = torch.device(device_str)
    est = load_estimator(
        lstm_hidden_size=lstm_hidden_size,
        lstm_num_layers=lstm_num_layers,
        n_mixtures=n_mixtures,
        filename=estimator_filename,
    ).to(device)
    typer.secho("-> generating plots", fg="yellow")
    generate_plots(
        est,
        sequence_length=seq_length,
        n_sequences=n_seq,
        bias=bias,
        rainbow=rainbow,
        normalize_filename=normalize_filename,
        linewidth=linewidth,
        base_plot_name=base_filename,
        device_str=device_str,
    )
    typer.secho("-> success!", fg="green")


if __name__ == "__main__":
    app()
