import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # Use GPU 1 for computation
import torch
import pandas as pd
import json
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline




token ="hf_***"
model_id = 'princeton-nlp/gemma-2-9b-it-SimPO'

df = pd.read_csv('/wi_dataset_completo.csv', sep=';', dtype=str)

generator = pipeline(
    "text-generation",
    model=model_id,
    device="cuda",
)

# prompt used to translate and extract key words form the original job posting
# in order to performe an optimal key words search
def prepare_prompt(job):
    chat = [
        { "role": "user", "content": f"""The following is a job posting scraped from the web. The text may contain grammatical errors, irrelevant details, or repetitions. Your task is to:

1. Clean the text by removing grammatical errors, irrelevant information, or duplicates.
2. Clean the job title and if there are typos, correct them. Moreover, if the "TITLE" contains unnecessary information like locations, job levels, or other extraneous details (e.g., "Consultants in Emergency Medicine - Doughiska"), remove those details and provide only the clean, concise job title (e.g., "Consultants in Emergency Medicine").
    - If the description is informative and the "TITLE" doesn't seem to match or accurately describe the job based on the "DESCRIPTION" provided, ignore the original title and instead generate a more suitable job title that better reflects the description.
    - If the description is missing, focus on the job title to generate the query. If the title is too vague or uninformative, focus on any available details from the description to create an accurate query.
3. Extract the key information from the job posting, such as job title, required skills, qualifications, and responsibilities.
4. Based on the extracted information, generate a **concise and accurate text query** that can be used to retrieve related job postings or information. This query will be used for a **keyword and basic concept search**.

**Note**: 
- In the query, exclude details related to job location, salary, and working schedule.

Please provide **only the final query** in English in a clear and organized format, ensuring it stays under 95 tokens. No explanation is needed.

Improve this posting and ensure the query is as optimized as possible.

**Job posting**:
{job}"""
            },
    ]    
    outputs = generator(chat,
                    do_sample=False,
                    eos_token_id=[generator.tokenizer.convert_tokens_to_ids("<end_of_turn>"), generator.tokenizer.eos_token_id],
                    max_new_tokens=100)

    return outputs[0]['generated_text'][1]['content']



aa='TITLE: '+df['title']+'\n\nDESCRIPTION: '+df['description'].apply(lambda x: str(x) if str(x)!='nan' else 'Description not provided')


path_saving = '/query_feature.jsonl'
path_progress = '/query_feature_progress.txt'

# Iterate through the dataset and save the cleaned features
diz = []
for i in tqdm(range(len(aa))[22602:]):
    id = df.iloc[i]['id']
    risposta = prepare_prompt(aa[i])
    diz.append({id: risposta})
    if (i) % 10 == 0:
        with open(path_progress, 'w', encoding='utf-8') as file:
            file.write(f"Job {i} of {len(df)}\n")
        if (i+1) == 1:
            with open(path_saving, 'w', encoding='utf-8') as file:
                for item in diz:
                    file.write(json.dumps(item, ensure_ascii=False) + '\n')
            diz = []
        else:
            with open(path_saving, 'a', encoding='utf-8') as file:
                for item in diz:
                    file.write(json.dumps(item, ensure_ascii=False) + '\n')
            diz = []
with open(path_saving, 'a', encoding='utf-8') as file:
    for item in diz:
        file.write(json.dumps(item, ensure_ascii=False) + '\n')  
