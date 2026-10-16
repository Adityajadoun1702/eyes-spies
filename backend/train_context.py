import argparse
import logging
import random
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torchvision.datasets import ImageFolder
from transformers import CLIPModel, CLIPProcessor

if __package__:
    from .context_model import CONTEXT_TASKS, ContextClassifier
else:
    from context_model import CONTEXT_TASKS, ContextClassifier

LOGGER = logging.getLogger("eyesspies.training")


class FolderSceneDataset(Dataset):
    def __init__(self, root: Path, processor: CLIPProcessor, task: str):
        self.images = ImageFolder(str(root))
        class_names = CONTEXT_TASKS[task]
        actual_names = {name.casefold() for name in self.images.classes}
        if actual_names != set(class_names):
            raise ValueError(
                f"{root} must contain exactly these class folders: "
                f"{', '.join(class_names)}. Found: {', '.join(self.images.classes)}"
            )
        if not self.images.samples:
            raise ValueError(f"No images found in {root}.")
        self.processor = processor
        self.class_indices = {
            self.images.class_to_idx[name]: class_names.index(name.casefold())
            for name in self.images.classes
        }

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image, source_label = self.images[index]
        pixels = self.processor(images=image, return_tensors="pt")["pixel_values"][0]
        return pixels, self.class_indices[source_label]


class HuggingFaceSceneDataset(Dataset):
    def __init__(self, dataset, processor: CLIPProcessor, task: str):
        self.dataset = dataset
        self.processor = processor
        labels = dataset.features["label"].names
        class_names = CONTEXT_TASKS[task]
        if {name.casefold() for name in labels} != set(class_names):
            raise ValueError(
                f"Dataset labels must be {', '.join(class_names)}; found {', '.join(labels)}."
            )
        self.label_indices = {
            index: class_names.index(name.casefold())
            for index, name in enumerate(labels)
        }

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        sample = self.dataset[index]
        image = sample["image"].convert("RGB")
        pixels = self.processor(images=image, return_tensors="pt")["pixel_values"][0]
        return pixels, self.label_indices[sample["label"]]


def load_datasets(args, processor):
    if args.dataset_id:
        if args.data_dir:
            raise ValueError("Use either --dataset-id or --data-dir, not both.")
        if args.task != "setting":
            raise ValueError(
                "The selected public dataset only provides indoor/outdoor labels. "
                "Use --data-dir for the urban/rural environment task."
            )
        try:
            from datasets import load_dataset
        except ImportError as exc:
            raise RuntimeError(
                "Install the training dependencies with: "
                "python -m pip install -r backend/training-requirements.txt"
            ) from exc

        LOGGER.info("Downloading/loading public dataset: %s", args.dataset_id)
        data = load_dataset(args.dataset_id, split="train")
        split = data.train_test_split(
            test_size=args.validation_fraction,
            seed=args.seed,
            stratify_by_column="label",
        )
        return (
            HuggingFaceSceneDataset(split["train"], processor, args.task),
            HuggingFaceSceneDataset(split["test"], processor, args.task),
        )

    if not args.data_dir:
        raise ValueError("Specify --dataset-id or --data-dir.")
    return (
        FolderSceneDataset(args.data_dir / "train", processor, args.task),
        FolderSceneDataset(args.data_dir / "validation", processor, args.task),
    )


def extract_features(encoder, dataset, args, device):
    batches = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
    )
    all_features = []
    all_labels = []
    with torch.inference_mode():
        for batch_index, (pixels, labels) in enumerate(batches, start=1):
            embeddings = encoder.get_image_features(pixel_values=pixels.to(device))
            embeddings = nn.functional.normalize(embeddings, dim=-1)
            all_features.append(embeddings.cpu())
            all_labels.append(labels)
            if batch_index % 50 == 0:
                LOGGER.info(
                    "Encoded %d/%d images",
                    min(batch_index * args.batch_size, len(dataset)),
                    len(dataset),
                )
    return torch.cat(all_features), torch.cat(all_labels)


def evaluate(classifier, head_name, features, labels):
    classifier.eval()
    head = getattr(classifier, head_name)
    with torch.inference_mode():
        logits = head(features)
        loss = nn.functional.cross_entropy(logits, labels)
        accuracy = (logits.argmax(dim=1) == labels).float().mean()
    return float(loss), float(accuracy)


def train(args):
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(
        "cuda" if torch.cuda.is_available() and args.device != "cpu" else "cpu"
    )
    LOGGER.info("Loading pretrained CLIP image encoder: %s", args.model)
    processor = CLIPProcessor.from_pretrained(args.model)
    encoder = CLIPModel.from_pretrained(args.model).to(device).eval()
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)

    train_data, validation_data = load_datasets(args, processor)
    LOGGER.info("Extracting frozen CLIP image features")
    train_features, train_labels = extract_features(encoder, train_data, args, device)
    validation_features, validation_labels = extract_features(
        encoder, validation_data, args, device
    )

    classifier = ContextClassifier(train_features.shape[-1])
    optimizer = torch.optim.AdamW(
        getattr(classifier, args.task).parameters(),
        lr=args.learning_rate,
        weight_decay=1e-4,
    )
    train_batches = DataLoader(
        TensorDataset(train_features, train_labels),
        batch_size=args.batch_size,
        shuffle=True,
    )
    best_accuracy = -1.0

    for epoch in range(1, args.epochs + 1):
        classifier.train()
        running_loss = 0.0
        head = getattr(classifier, args.task)
        for features, labels in train_batches:
            logits = head(features)
            loss = nn.functional.cross_entropy(logits, labels)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            running_loss += float(loss.detach()) * len(labels)

        validation_loss, validation_accuracy = evaluate(
            classifier, args.task, validation_features, validation_labels
        )
        LOGGER.info(
            "Epoch %d/%d: train_loss=%.4f validation_loss=%.4f validation_accuracy=%.3f",
            epoch,
            args.epochs,
            running_loss / len(train_features),
            validation_loss,
            validation_accuracy,
        )
        if validation_accuracy > best_accuracy:
            best_accuracy = validation_accuracy
            save_checkpoint(args, classifier, train_features.shape[-1], validation_accuracy)


def save_checkpoint(args, classifier, embedding_size, validation_accuracy):
    args.output.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model_id": args.model,
        "embedding_size": embedding_size,
        "tasks": {},
        "state_dict": {},
    }
    if args.output.is_file():
        existing = torch.load(args.output, map_location="cpu", weights_only=True)
        if (
            existing.get("model_id") != args.model
            or existing.get("embedding_size") != embedding_size
        ):
            raise ValueError(
                "Existing context checkpoint uses a different CLIP model or embedding size."
            )
        checkpoint = existing

    prefix = f"{args.task}."
    checkpoint["state_dict"].update({
        key: value.detach().cpu()
        for key, value in classifier.state_dict().items()
        if key.startswith(prefix)
    })
    checkpoint["tasks"][args.task] = {
        "labels": list(CONTEXT_TASKS[args.task]),
        "validation_accuracy": validation_accuracy,
        "seed": args.seed,
    }
    torch.save(checkpoint, args.output)
    LOGGER.info("Saved %s classifier checkpoint to %s", args.task, args.output)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train one scene-context head on labeled images using frozen CLIP features."
    )
    parser.add_argument("--task", choices=tuple(CONTEXT_TASKS), required=True)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--dataset-id")
    parser.add_argument(
        "--model",
        default="openai/clip-vit-base-patch32",
        help="Hugging Face CLIP model ID; must match the API's CLIP_MODEL.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).parent / "models" / "context_head.pt",
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--device", choices=("auto", "cpu"), default="auto")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.learning_rate <= 0:
        parser.error("epochs and batch size must be positive and learning rate must exceed zero")
    if not 0 < args.validation_fraction < 1:
        parser.error("validation fraction must be between zero and one")
    return args


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    train(parse_args())
