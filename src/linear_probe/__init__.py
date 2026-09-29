"""Linear probe training and inference for the modular pipeline."""


# Imported lazily so the embedding-cache probe (``embedding_probe``) works without
# pytorch_lightning installed.
def train_main(*args, **kwargs):
    from .train import main

    return main(*args, **kwargs)


def predict_main(*args, **kwargs):
    from .predict import main

    return main(*args, **kwargs)


__all__ = ["train_main", "predict_main"]
