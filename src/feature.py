import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # Use GPU 1 for computation
import torch
import pandas as pd
import json
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# Replace with your Hugging Face Token for access (ensure you have the appropriate permissions)



token ="hf_*****"
# model to perform summarization
model_id = 'princeton-nlp/gemma-2-9b-it-SimPO'

df = pd.read_csv('/wi_dataset_completo.csv', sep=';', dtype=str)

generator = pipeline(
    "text-generation",
    model=model_id,
    device="cuda",
)



# prompt used to translate and extract key information form the original job posting
def prepare_prompt(job):
    chat = [
        { "role": "user", "content": 
            f"""You are an AI assistant tasked with extracting the most important features from job postings to aid in classification according to specific occupational categories according to ISCO (International Standard Classification of Occupations). 
Your goal is to identify key details that are crucial for determining the appropriate classification.
            
Please clean and summarize the following job posting scraped from the internet. Focus on extracting only the essential information, removing any unnecessary elements such as headers, footers, contact details, irrelevant formatting, or repetitive text.

Make sure to discard any marketing language, unnecessary formatting, and repetitive or irrelevant sections. Provide the cleaned and summarized content in a discursive way. Ensuring that all extracted information is in English and in no more than 200 tokens.

**Note**: 
- Exclude details related to job location, salary, and working schedule.
- **Report the job title separately**, and if there are typos, correct them. Moreover, if the "ORIGINAL TITLE" contains unnecessary information like locations, job levels, or other extraneous details (e.g., "Consultants in Emergency Medicine - Doughiska"), remove those details and provide only the clean, concise job title (e.g., "Consultants in Emergency Medicine").
- If the description is informative and the "ORIGINAL TITLE" doesn't seem to match or accurately describe the job based on the "DESCRIPTION" provided, ignore the original title and instead generate a more suitable job title that better reflects the description.

**Output format**:
**Job Title**: [Clean job title here]
**Description**: [Summarized description here]

Job posting:
{job}"""
            },
    ]    
    outputs = generator(chat,
                    do_sample=False,
                    eos_token_id=[generator.tokenizer.convert_tokens_to_ids("<end_of_turn>"), generator.tokenizer.eos_token_id],
                    max_new_tokens=200)

    return outputs[0]['generated_text'][1]['content']



# built a unique block of text for each job posting with title and description
aa='TITLE: '+df['title']+'\n\nDESCRIPTION: '+df['description'].apply(lambda x: str(x) if str(x)!='nan' else 'Description not provided')


path_saving = '/feature.jsonl'
path_progress = '/feature_progress.txt'



# Iterate through the dataset and save the cleaned features
diz = []
for i in tqdm(range(len(aa))[20321:]):
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
