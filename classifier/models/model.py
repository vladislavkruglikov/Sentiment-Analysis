import torch

from typing import Dict

from allennlp.data import Vocabulary
from allennlp.data.fields.text_field import TextFieldTensors
from allennlp.models import Model
from allennlp.training.metrics.categorical_accuracy import CategoricalAccuracy
from allennlp.modules.text_field_embedders.text_field_embedder import TextFieldEmbedder
from allennlp.modules.seq2vec_encoders import Seq2VecEncoder
from allennlp.nn import util


@Model.register("classifier")
class SentimentModel(Model):
    def __init__(self,
                 vocab: Vocabulary,
                 embedder: TextFieldEmbedder,
                 encoder: Seq2VecEncoder):
        super().__init__(vocab)
        self.embedder = embedder
        self.encoder = encoder
        num_labels = vocab.get_vocab_size("labels")
        self.classifier = torch.nn.Linear(self.encoder.get_output_dim(), num_labels)
        self.accuracy = CategoricalAccuracy()

    def forward(
        self, text: TextFieldTensors, label: torch.Tensor = None
    ) -> Dict[str, torch.Tensor]:
        embedded_text = self.embedder(text)
        mask = util.get_text_field_mask(text)
        encoded_text = self.encoder(embedded_text, mask)
        logits = self.classifier(encoded_text)
        probs = torch.nn.functional.softmax(logits)
        loss = torch.nn.functional.cross_entropy(logits, label)

        self.accuracy(logits, label)

        return {'loss': loss, 'probs': probs}

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {"accuracy": self.accuracy.get_metric(reset)}
