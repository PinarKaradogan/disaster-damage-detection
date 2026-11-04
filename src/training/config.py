from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class TrainingConfig:
    model_name: str

    seed: int = 42
    input_size: int = 128
    num_classes: int = 4

    batch_size: int = 32
    smoke_batch_size: int = 8
    num_workers: int = 0

    epochs: int = 30
    early_stopping_patience: int = 5

    optimizer: str = "adamw"
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4

    scheduler: str = "cosine"
    scheduler_eta_min: float = 1e-6

    amp: bool = True
    training_mode: str = "full"

    def __post_init__(self):
        if self.training_mode not in {"full", "head_only"}:
            raise ValueError("training_mode must be full or head_only")
        if self.training_mode == "head_only" and self.model_name != "resnet34_early_fusion":
            raise ValueError("head_only training requires resnet34_early_fusion")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
