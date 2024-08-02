from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from multiprocessing import Process
import torch
import pandas as pd
import math
from tqdm import tqdm

class Llama2Bot:
    def __init__(self, device_id=0):
        self.model_name="meta-llama/Llama-2-7b-chat-hf"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name)

        if torch.cuda.is_available():
            self.model = self.model.to(f'cuda:{device_id}')

        self.pipe = pipeline(task="text-generation", model=self.model, tokenizer=self.tokenizer, max_length=4096, device=device_id)
        

    def respond(self, question): 
        result = self.pipe(f"<s>[INST] {question} [/INST]")
        q_a = result[0]["generated_text"]
        answer = q_a.split('[/INST]')[1]
        return answer

def asking_questions_csv(bot, csv_file):
    df = pd.read_csv(csv_file)
    ask_index = 0
    for ind in df.index:
        answer = df['answer'][ind]
        try:
            x = math.isnan(answer)
            break
        except:
            ask_index += 1

    with tqdm(total=len(df)) as pbar:
        pbar.update(ask_index)
        while ask_index < len(df):
            question = df.loc[ask_index, "question"]
            ans = bot.respond(question)
            df.loc[ask_index, "answer"] = ans
            ask_index += 1
            pbar.update(1)
            df.to_csv(csv_file, index=False)

if __name__ == "__main__":
    csv_file="questions/25-MR5_allsome/MR5_allsome_pair.csv"
    bot = Llama2Bot(device_id=0)
    asking_questions_csv(bot, csv_file)
