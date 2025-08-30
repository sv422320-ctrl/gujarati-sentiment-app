import torch.nn as nn
from transformers import AutoModel

class BertCNN(nn.Module):
    def __init__(self, model_name, num_labels):
        super(BertCNN, self).__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.conv1 = nn.Conv1d(in_channels=self.bert.config.hidden_size, out_channels=256, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(256, num_labels)

    def forward(self, input_ids=None, attention_mask=None, labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state  # [batch, seq_len, hidden]
        x = last_hidden_state.permute(0, 2, 1)  
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x).squeeze(-1)  # [batch, out_channels]
        x = self.dropout(x)
        logits = self.fc(x)

        return {"logits": logits}
