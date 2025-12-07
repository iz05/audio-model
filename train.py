import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn import CrossEntropyLoss
from transformers import AutoTokenizer, AdamW, get_scheduler
from audio_feature_model import QWenLMHeadModelWithFeatures
import pandas as pd
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig
from tokenization_qwen import QWenTokenizerRawAudio
import os

DEVICE = "cuda"
PRETRAINED_MODEL = "Qwen/Qwen-Audio"
BATCH_SIZE = 1
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

        try:
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

        except Exception as e:
            print(e)

# --- 2. Load tokenizer and dataset ---
tokenizer = QWenTokenizerRawAudio.from_pretrained(PRETRAINED_MODEL, trust_remote_code=True)
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
losses = []
save_dir = '/home/ixzhu/Qwen-Audio/checkpoints'

model.train()
for epoch in range(EPOCHS):
    print(f"Epoch {epoch+1}/{EPOCHS}")
    for batch in tqdm(dataloader):
        
        try:
            item = batch[0]

            input_ids = item["input_ids"].unsqueeze(0).to(DEVICE)  # add batch dim
            attention_mask = item["attention_mask"].unsqueeze(0).to(DEVICE)
            labels = item["labels"].unsqueeze(0).to(DEVICE)
            audio_info = item["audio_info"]

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels, audio_info=audio_info)
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            losses.append(loss.item())
        except Exception as e:
            print(e)
    
    checkpoint_path = os.path.join(save_dir, f"checkpoint_epoch_{epoch+1}")
    model.save_pretrained(checkpoint_path)
    tokenizer.save_pretrained(checkpoint_path)
    torch.save(optimizer.state_dict(), os.path.join(checkpoint_path, "optimizer.pt"))
    torch.save(scheduler.state_dict(), os.path.join(checkpoint_path, "scheduler.pt"))
    torch.save(losses, os.path.join(checkpoint_path, "losses.pt"))
    print(f"Saved checkpoint at end of epoch {epoch+1}")

    print(f"Epoch {epoch+1} finished. Last batch loss: {loss.item():.4f}")

# --- 6. Save fine-tuned model ---
final_path = os.path.join(save_dir, f"trained_model")
model.save_pretrained(final_path)
tokenizer.save_pretrained(final_path)
torch.save(losses, os.path.join(final_path, "losses.pt"))
print("Training complete. Model saved.")
