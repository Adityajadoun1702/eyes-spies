import torch
from torch import nn

CONTEXT_TASKS = {
    "setting": ("indoor", "outdoor"),
    "environment": ("urban", "rural"),
}


class ContextClassifier(nn.Module):
    def __init__(self, embedding_size: int):
        super().__init__()
        self.setting = nn.Linear(embedding_size, len(CONTEXT_TASKS["setting"]))
        self.environment = nn.Linear(
            embedding_size, len(CONTEXT_TASKS["environment"])
        )


def context_from_probabilities(
    setting_scores: torch.Tensor, environment_scores: torch.Tensor
) -> dict:
    setting_index = int(torch.argmax(setting_scores))
    environment_index = int(torch.argmax(environment_scores))
    return {
        "setting": ["Indoor", "Outdoor"][setting_index],
        "setting_confidence": round(float(setting_scores[setting_index]), 4),
        "environment": ["Urban", "Rural"][environment_index],
        "environment_confidence": round(float(environment_scores[environment_index]), 4),
    }
