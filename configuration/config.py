from dataclasses import dataclass, field


@dataclass
class Config:
    framework: str = field(
        default="FCL",
        metadata={"help": "Framework to evaluate [CL]"},
    )
    dir_data: str = field(
        default="./data/",
        metadata={"help": "Directory to save the datasets"},
    )
    dir_output: str = field(
        default="./output/",
        metadata={"help": "Directory to save the output"},
    )
    dataset_name: str = field(
        default="cifar10",
        metadata={"help": "Name of the dataset"},
    )
    model_name: str = field(
        default="default",
        metadata={"help": "Model architecture [mlp, resnet, default]"},
    )
    batch_size: int = field(
        default=10,
        metadata={"help": "Batch size"},
    )
    lr: float = field(
        default=0.1,
        metadata={"help": "Learning rate"},
    )
    optimizer: str = field(
        default="sgd",
        metadata={"help": "Optimizer [sgd, adam]"},
    )
    local_epochs: int = field(
        default=3,
        metadata={
            "help": "Number of epochs at batch-level (multiple gradient updates per batch)"
        },
    )
    n_runs: int = field(
        default=3,
        metadata={"help": "Number of runs for each experiment"},
    )
    n_tasks: int = field(
        default=-1,
        metadata={
            "help": "Number of tasks (-1 default number of tasks for each dataset)"
        },
    )
    with_memory: int = field(
        default=1,
        metadata={"help": "Use of episodic memory [0 (no), 1 (yes)]"},
    )
    memory_size: int = field(
        default=500,
        metadata={"help": "Size of episodic memory [200, 500, 1000]"},
    )
    update_strategy: str = field(
        default="balanced",
        metadata={"help": "Memory update strategy [reservoir, balanced]"},
    )
    sampling_strategy: str = field(
        default="random",
        metadata={"help": "Memory sampling strategy [random, uncertainty]"},
    )
    balanced_update: str = field(
        default="uncertainty",
        metadata={
            "help": "Update strategy for class-balanced memory management [random, uncertainty]"
        },
    )
    uncertainty_score: str = field(
        default="bregman",
        metadata={
            "help": "Uncertainty metric for uncertainty management (bregman, confidence, margin, entropy, rainbow)"
        },
    )
    subsample_size: int = field(
        default=50,
        metadata={
            "help": "Size of the subsample (taken from the memory for replay) for computing the uncertainty scores when sampling from the memory"
        },
    )
    balanced_step: str = field(
        default="bottomk",
        metadata={
            "help": "Sampling strategy for uncertainty-based class-balanced memory management (step (step-sized), topk (top-k), bottomk (bottom-k))"
        },
    )
    n_clients: int = field(
        default=5,
        metadata={"help": "Number of clients"},
    )
    ####################### FEDERATED PARAMETERS ###########################
    overlap: str = field(
        default="overlap",
        metadata={"help": "Overlapping tasks across clients (overlap, non-overlap)"},
    )
    burnin: int = field(
        default=30,
        metadata={
            "help": "Burnin time (number of epochs) before a client contributes to the communication rounds"
        },
    )
    jump: int = field(
        default=5,
        metadata={
            "help": "Number of epochs to jump before a client contributes to the communication rounds"
        },
    )
    fl_update: str = field(
        default="w_favg",
        metadata={
            "help": "Memory sampling strategy [favg (FedAvg), w_favg (weighted FedAvg), fprox (FedProx), w_fprox (weighted fprox)]"
        },
    )
    mu: float = field(
        default=0.01,
        metadata={"help": "Parameter for proximal term in FedProx"},
    )
