from openai import OpenAI
import pandas as pd
import math
from tqdm import tqdm
import os
os.environ['OPENAI_API_KEY'] = 'YOUR_OPENAI_API_KEY'

client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

def get_answer(question):
  completion = client.chat.completions.create(
    model="gpt-3.5-turbo-0125",
    messages=[
      {"role": "user", "content": question}
    ],
      max_tokens=50
  )
  answer = completion.choices[0].message.content
  return answer

def asking_questions_csv(csv_file):
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
          ans = get_answer(question)
          df.loc[ask_index, "answer"] = ans
          ask_index += 1
          pbar.update(1)
          df.to_csv(csv_file, index=False)


asking_questions_csv("25-MR5_allsome/MR5_allsome_pair.csv")
