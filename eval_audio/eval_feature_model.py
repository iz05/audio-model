from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig
import torch
from tqdm import tqdm
from audio_feature_model import QWenLMHeadModelWithFeatures
from tokenization_qwen import QWenTokenizerRawAudio

torch.manual_seed(1234)

PRETRAINED_MODEL = "Qwen/Qwen-Audio"
DEVICE = 'cuda'

# load tokenizer
tokenizer = QWenTokenizerRawAudio.from_pretrained(PRETRAINED_MODEL, trust_remote_code=True)
tokenizer.pad_token_id = tokenizer.eod_id
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# load model
MODEL_TYPE = "CNN"
save_dir = f'/home/ixzhu/Qwen-Audio/checkpoints_{MODEL_TYPE}/checkpoint_epoch_5'
model = QWenLMHeadModelWithFeatures.from_pretrained(save_dir, device_map="cuda", trust_remote_code=True).eval()
model.eval()

# load json and get questions
import json

json_path = "eval_audio/data/aqa/clothoaqa_eval.jsonl"

unique_qs = {}
with open(json_path, "r") as f:
    for line in f:
        data = json.loads(line)
        key = ("eval_audio/data/aqa/" + data["audio"], data["question"])
        if key not in unique_qs:
            unique_qs[key] = []
        unique_qs[key].append(data["gt"])  # store all gts for reference


# Specify hyperparameters for generation (No need to do this if you are using transformers>4.32.0)
# model.generation_config = GenerationConfig.from_pretrained("Qwen/Qwen-Audio", trust_remote_code=True)

results = []

for (audio_url, question), gts in tqdm(unique_qs.items(), desc='Evaluating'):
    query = f'<audio>{audio_url}</audio><|startofanalysis|><|en|><|question|>{question}<|answer|>'
    prefix_length = len(query)

    audio_info = tokenizer.process_audio(query)
    inputs = tokenizer(query, return_tensors='pt', audio_info=audio_info)
    inputs = inputs.to(model.device)

    pred = model.generate(**inputs, 
        audio_info=audio_info,
        do_sample=False,
        max_new_tokens=4,
        min_new_tokens=1,
        length_penalty=1.0,
        num_return_sequences=1,
        repetition_penalty=1.0,
        use_cache=False,
        pad_token_id=tokenizer.eod_id,
        eos_token_id=tokenizer.eod_id,
    )
    print("Query: ", query)
    response = tokenizer.decode(pred.cpu()[0], skip_special_tokens=False,audio_info=audio_info)
    print("Raw response: ", response)
    filtered_response = response[prefix_length:].strip().split("<|endoftext|>")[0].strip()
    print("Filtered response: ", filtered_response, " Actual responses: ", gts)
    results.append({
        'audio': audio_url,
        'question': question,
        'gt': gts,
        'response': filtered_response
    })

out_path = f"/home/ixzhu/Qwen-Audio/eval_audio/results/eval_results_{MODEL_TYPE}.json"
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)

print(f"Saved {len(results)} items to {out_path}")