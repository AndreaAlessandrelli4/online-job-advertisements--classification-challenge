import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # Use GPU 1 for computation
import torch
import json
from tqdm import tqdm
import pandas as pd
import random
import time
import re
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline




token ="hf_*****"
# model to perform classification
model_id = 'princeton-nlp/gemma-2-9b-it-SimPO'


generator = pipeline(
    "text-generation",
    model=model_id,
    device="cuda",
)



# check list to avoid allucination in the 4-digit ISCO 'code'
labels= pd.read_csv('/wi_labels.csv', dtype=str)
check_list = list(labels['code'])






# loading of cleaned job posting in data_text_diz where each key is the unique ID of the job posting
data_text=[]
with open('/feature.jsonl', 'r', encoding='utf-8') as file:
    for f in file:
        data_text.append(json.loads(f))
data_text_diz = {}
for t in data_text:
    k,v = list(t.items())[0]
    data_text_diz[k]=v 
    



# loading of the materials retieved from Weaviate in data where each key is the unique ID of the job posting
files=os.listdir('/v10')
jsonl_files = [f for f in files if f.endswith('.jsonl')]
len(jsonl_files)
data = []
for f in jsonl_files:
    with open(f'/{f}','r', encoding='utf-8') as file:
        for f in file:
            data.append(json.loads(f))
            
            

# function that takes as INPUT: 
#   "ID" of the job posting, 
#   "data"(the material retrieved from weaviate) 
#   "data_text_diz"(the cleaning job posting) 
# and gives as output the prompt to pass to ChatGPT
def get_query(idx, data=data, data_text_diz=data_text_diz):
    temp = data[idx]
    id = temp['id']
    labels = temp['codici']
    categories = temp['testo']
    job = data_text_diz[id]
    query=f"""You are an AI assistant trained to classify job postings according to ISCO (International Standard Classification of Occupations) categories. 
You will be provided with the job posting that needs to be classified, along with official descriptions of various ISCO occupations that are similar to the job posting. 
Follow these steps to perform the classification:

1. Analyze the "JOB POSTING":
    - Carefully read the job posting that needs to be classified.

2. Examine the "ISCO Descriptions":
    - Review the provided ISCO occupation descriptions, focusing on the main responsibilities and tasks associated with each category.

3. "Consider Alternative Labels":
    - Pay special attention to the label and alternative labels used within each ISCO description.
    - If you find any **literal** or **semantic** matches between the job title in the posting and the label and alternative labels listed in the ISCO descriptions, prioritize that ISCO category.

4. "Classification":
    - Based on the analysis of the responsibilities, tasks, and potential matches in alternative labels, select the most appropriate ISCO category for the job posting.
    - Provide only the label of the selected ISCO category as your final answer. Example of output: $$ISCO category$$.

5. "Fallback to General Knowledge":
    - If you are unable to find a relevant match from the provided ISCO descriptions, or if you are uncertain about which code to choose based on the materials provided, use your own knowledge to classify the job posting.
    - However, only use this approach as a **last resort** when the job posting does not fit any of the provided ISCO categories at all.
    - Even in this case, ensure that the classification is reasonable and closely related to the responsibilities and tasks described in the job posting.

"JOB POSTING":
{job}

"ISCO Descriptions":
{categories}
------------------------------

Please provide for the given job posting only one of the following ISCO codes without any explanation. If no suitable code is found, you may use your own knowledge as a fallback, but only if the job posting is clearly unrelated to the ISCO descriptions provided:
{labels}
"""
    return id, query



# prompt used to translate and extract key information form the original job posting
def prepare_prompt(chat):
    """ chat = [
        { "role": "user", "content": text_prompt},
    ]   """  
    outputs = generator(chat,
                    do_sample=False,
                    eos_token_id=[generator.tokenizer.convert_tokens_to_ids("<end_of_turn>"), generator.tokenizer.eos_token_id],
                    max_new_tokens=20)

    return outputs[0]['generated_text'][1]['content']



saving_path='/class_chat.jsonl'


# loop over each job ID
# for each job ID we use "gemma-2-9b-it-SimPO" for the classification and we save the output in a jsonl file
diz =[]
for idx in tqdm(range(len(data))):
    id, qq=get_query(idx)
    try:
        mex = [{"role": "user", "content": qq}]
        response=prepare_prompt(mex)
        class1 = re.findall(r'\d{4}',response)[0]
        if class1 in check_list:
            diz.append({id:class1})
        else:
            mex = [{"role": "user", "content": qq}, {"role": "assistant", "content": class1},{"role": "user", "content": f"The given ISCO code dose not exist, please be sure that the ISCO code is in the following list:\n{check_list}"}]
            response=prepare_prompt(mex)
            class1 = re.findall(r'\d{4}',response)[0]
            if class1 in check_list:
                diz.append({id:class1})
            else:
                diz.append({id:'-1'})
                print('missing')
    except:
        diz.append({id:'-1'})
        print('missing')
        
        
    if idx%10==0:
        if (idx+1)==1:
            with open(saving_path, 'w', encoding='utf-8') as file:
                for d in diz:
                    file.write(json.dumps(d, ensure_ascii=False) + '\n')
        else:
            with open(saving_path, 'a', encoding='utf-8') as file:
                for d in diz:
                    file.write(json.dumps(d, ensure_ascii=False) + '\n')
        diz=[]
with open(saving_path, 'a', encoding='utf-8') as file:
    for d in diz:
        file.write(json.dumps(d, ensure_ascii=False) + '\n')
