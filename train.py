import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn import CrossEntropyLoss
from transformers import AutoTokenizer, AdamW, get_scheduler
from audio_feature_model import QWenLMHeadModelWithFeatures
import pandas as pd
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig

DEVICE = "cuda"
PRETRAINED_MODEL = "Qwen/Qwen-Audio"
BATCH_SIZE = 2
LR = 1e-4
EPOCHS = 3
MAX_SEQ_LEN = 512  # adjust if needed

AUDIO_ROOT = "/home/ixzhu/Qwen-Audio/eval_audio/data/aqa/clothoqa/audio_files/"

# --- 1. Dataset ---
class AudioQADataset(Dataset):
    def __init__(self, csv_path, tokenizer):
        self.df = pd.read_csv(csv_path)
        self.tokenizer = tokenizer
        self.data = []
        for _, row in self.df.iterrows():
            audio_file = AUDIO_ROOT + row['file_name']
            question = row['QuestionText']
            answer = row['answer']

            # same prompt format as eval
            prompt = f"<audio>{audio_file}</audio><|startofanalysis|><|en|><|question|>{question}<|answer|>"
            self.data.append((prompt, answer, audio_file))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        prompt, answer, audio_file = self.data[idx]

        # process audio for the tokenizer
        audio_info = self.tokenizer.process_audio(prompt)
        inputs = self.tokenizer(prompt, return_tensors='pt', audio_info=audio_info)
        input_ids = inputs['input_ids'].squeeze(0)
        attention_mask = inputs['attention_mask'].squeeze(0)

        # tokenize answer
        labels = self.tokenizer(answer, return_tensors='pt').input_ids.squeeze(0)

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
            'audio_info': audio_info
        }

# --- 2. Load tokenizer and dataset ---
tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_MODEL, trust_remote_code=True)
tokenizer.pad_token_id = tokenizer.eod_id
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
dataset = AudioQADataset("/home/ixzhu/Qwen-Audio/eval_audio/data/aqa/clotho_aqa_train.csv", tokenizer)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=lambda x: x)

# --- 3. Load pretrained model and initialize with features ---
pretrained_base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-Audio", device_map="cuda", trust_remote_code=True).eval()
config = pretrained_base.config
print("Initializing model...")
model = QWenLMHeadModelWithFeatures(config).to(DEVICE)
print("Finished intializing model.")

# Load pretrained weights into matching parameters
print("Transferring weights...")
pretrained_dict = pretrained_base.state_dict()
model_dict = model.state_dict()
pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
model_dict.update(pretrained_dict)
model.load_state_dict(model_dict)
print("Finished transferring weights.")

# Freeze everything except audio_feature_project
print("Freezing weights...")
for name, param in model.named_parameters():
    param.requires_grad = "audio_feature_project" in name
print("Finished freezing weights.")

# --- 4. Optimizer & scheduler ---
optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)
num_training_steps = EPOCHS * len(dataloader)
scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)
loss_fn = CrossEntropyLoss()

# --- 5. Training loop ---
model.train()
for epoch in range(EPOCHS):
    print(f"Epoch {epoch+1}/{EPOCHS}")
    for batch in tqdm(dataloader):
        # collate batch manually
        pad_id = tokenizer.pad_token_id
        input_ids = torch.nn.utils.rnn.pad_sequence(
            [b['input_ids'] for b in batch],
            batch_first=True,
            padding_value=pad_id
        ).to(DEVICE)
        attention_mask = torch.nn.utils.rnn.pad_sequence(
            [b['attention_mask'] for b in batch],
            batch_first=True,
            padding_value=0
        ).to(DEVICE)
        labels = torch.nn.utils.rnn.pad_sequence(
            [b['labels'] for b in batch],
            batch_first=True,
            padding_value=-100
        ).to(DEVICE)
        audio_info = [b['audio_info'] for b in batch]


        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels, audio_info=audio_info)
        loss = outputs.loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

    print(f"Epoch {epoch+1} finished. Last batch loss: {loss.item():.4f}")

# --- 6. Save fine-tuned model ---
model.save_pretrained("qwen_audio_mlp_finetuned")
tokenizer.save_pretrained("qwen_audio_mlp_finetuned")
print("Training complete. Model saved.")
