def cycle_dataloader(dl):
    while True:
        for data in dl:
            yield data


def divisible_by(numer, denom):
    return (numer % denom) == 0
