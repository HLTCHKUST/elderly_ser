import torch
import torch.nn as nn




class EmotionClassifier(nn.Module):
    def __init__(self, n_train_steps, n_classes, do_prob, model):
        super(EmotionClassifier, self).__init__()
        self.model = model
        self.dropout = nn.Dropout(do_prob)
        self.out = nn.Linear(768, n_classes)
        self.n_train_steps = n_train_steps
        self.step_scheduler_after = "batch"

    def forward(self, ids, mask):
        output_1 = self.model(ids, attention_mask=mask)
        output_2 = self.dropout(output_1)
        output = self.out(output_2)
        return output