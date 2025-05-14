import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Use GPU 0 for computation
import torch
import numpy as np
import json
import re
import pandas as pd
import weaviate
import unicodedata
from weaviate.gql.get import HybridFusion
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from FlagEmbedding import FlagReranker


key = 'caq***'
url = "https://***"


# model to perform vectorization
model_rag = SentenceTransformer('all-mpnet-base-v2', device="cuda")
# model to perform reranking
reranker = FlagReranker('BAAI/bge-reranker-large')

# client Weaviate
client = weaviate.Client(
    url=url,  
    auth_client_secret=weaviate.auth.AuthApiKey(api_key=key), 
)


# list of possible 4-digit code, used for checking the retrieved ones
df_original = pd.read_csv("/wi_labels.csv", dtype=str)
check_codes = list(set(df_original['code']))
del df_original


# function to transform text into semantic vectors
def generate_embeddings(entry, model=model_rag):
        embedding = model.encode(entry, convert_to_numpy=True)
        return embedding


# function to remove unicode from strings
def remove_unicode(stringa):
        stringa_normalizzata = unicodedata.normalize('NFKD', stringa)
        stringa = stringa_normalizzata.encode('ascii', 'ignore').decode('ascii')
        return stringa.replace("////","-").replace("//","-").replace("/","-")


# functions used to clean job decription 
# function1
def split_sections(lines):
    sections = []
    section = []
    for line in lines:
        if re.match(r"-", line) or not section:
            section.append(line)
        else:
            sections.append(section)
            section = [line]
    if section:
        sections.append(section)
    return sections

# function2
def clean_job_description(text):
    text=text.replace('#### ','')
    text=text.replace('### ','')
    text=text.replace('## ','')
    text=text.replace('# ','')
    text = re.sub(r"\*\*.*?\*\*", "", text)
    text = re.sub(r"\*.*?\*", "", text)
    text = text.strip()
    lines = text.split('\n')

    clean_lines = [line.strip() for line in lines if line.strip()]
    clean_lines = [line.strip() for line in lines if 'No information is available' not in line]
    final_text = "\n\n".join("\n".join(section) for section in split_sections(clean_lines))

    return re.sub(r'\n+','\n',final_text.replace(':','').replace('&','and').replace('/','-').strip()).strip()



# properties where to search and weight of the searching process
x = [
'iSCO_desc', 
'iSCO_label^2',  
'occup_alt_label^2', 
'occup_desc', 
'occup_label^2'
]

# preperies to give as output of the searching process
y = ['iSCO_code', 
'iSCO_desc', 
'iSCO_label', 
'knowledge', 
'knowledge_essential', 
'knowledge_optional', 
'occup_alt_label', 
'occup_code', 
'occup_desc', 
'occup_label', 
'skills', 
'skills_essential', 
'skills_optional',
"text"]



# function which perform a search of weaviate database 
#dd: key word feature estracted from original job posting,
#dd_text: textual feature extraction from original job posting
#alpha [vector]: kinf of search 0:only keyword, 1:only vector search
#num [vector]: number of document retrieved by dd
#num_tex: number of document retrieved by dd_text
def wea_complete_occ(dd,dd_text, alpha, num, num_text,x, y):
    dd_cleaned_list = clean_job_description(dd).replace('-','').replace("\\\\",'-').replace("\\",'-').replace("\\\\\\",'-').replace("//",'-').replace("/",'-').replace('&','and')
    dd_cleaned_text = dd_text.replace('-','').replace("\\\\",'-').replace("\\",'-').replace("\\\\\\",'-').replace("//",'-').replace("/",'-')
    codes = []
    testo_RAG = {}

    for i, j in zip(alpha, num):
        if type(x) != list:
            x = [x]
        
        nome_collection='Descrizioni_occupazioni_en'
        response = (
                    client.query
                    .get(nome_collection, list(y))
                    .with_hybrid(
                                    query= dd_cleaned_list,#', '.join(extract_important_tokens(domanda)),
                                    vector=list(generate_embeddings(dd_cleaned_list)),#                        ----> la query è fatta solo sul testo della DOMANDA
                                    properties= list(x), ## where to search & boost factors           ----> c'è un boost su ARTICOLO e CODICE
                                    alpha= i,       # 1 = vector, 0 = keyword                    ----> parametro che faremo variare
                                    fusion_type=HybridFusion.RELATIVE_SCORE, # otherwise it operates on ranks
                                )
                    .with_limit(j)
                    .do()
                )
        try:
            response_hybrid = response['data']['Get'][nome_collection]
        except:
            response_hybrid = []
        response = (
                client.query
                .get(nome_collection, list(y))
                .with_bm25(
                                query=dd_cleaned_text,
                                properties= list(x),
                            )
                .with_limit(num_text)
                .do()
            )
        try:
            response_key = response['data']['Get'][nome_collection]
        except:
            response_key = []
             
        response = (
                client.query
                .get(nome_collection, list(y))
                .with_bm25(
                                query=dd_cleaned_text,
                                properties= ['iSCO_label',  'occup_alt_label^2', 'occup_label^2'],
                            )
                .with_limit(3)
                .do()
            )
        try:
            response_text= response['data']['Get'][nome_collection]
        except:
            response_text = []
        
        #response_tot vector with all occupations retrieved
        response_tot = response_hybrid + response_key 
        response_tot += response_text
        
        # we build a dict for each occupations retrieved where each key is the occupation unique code
        for item in response_tot:
            ISCO_code = item["iSCO_code"]
            occ_code = item["occup_code"]
            if occ_code=='NONE':
                testo_RAG[ISCO_code] = "ISCO Category: " + item["iSCO_label"] + "\nISCO category code: " + item["iSCO_code"]+ "\nCategory Description: " + item["iSCO_desc"] 

            else:
                testo_RAG[occ_code] = "ISCO Category: " + item["iSCO_label"] + "\nISCO category code: " + item["iSCO_code"]+ "\nCategory Description: " + item["iSCO_desc"] 
                testo_RAG[occ_code] = testo_RAG[occ_code] + "\nOccupations in this category that might be related to the job ad:\n"
                testo_RAG[occ_code] = testo_RAG[occ_code] + "\nOccupation Title: " + item["occup_label"]  + "\nOccupation Description: " + item["occup_desc"] #+ "\nOccupation Alternative Titles: " + item["occup_alt_label"].replace("\n", ", ")

            codes.append(item['occup_code'])
        codes_finali = []

    for i in list(set(codes)):
        if i[0:4] in check_codes:
            codes_finali.append(i)
    return list(set(codes_finali)), testo_RAG





        

# loading of cleaned job posting in data_text_diz where each key is the unique ID of the job posting
data_text=[]
with open('/feature.jsonl', 'r', encoding='utf-8') as file:
    for f in file:
        data_text.append(json.loads(f))
data_text_diz = {}
for t in data_text:
    k,v = list(t.items())[0]
    data_text_diz[k]=v  

 



    
    
# load of structurated features extraction from original job postings
listati = []
with open('/old_feature.jsonl','r', encoding = 'utf-8') as file:
    for f in file:
        listati.append(json.loads(f))
listati_diz = {}
for t in listati:
    k,v = list(t.items())[0]
    listati_diz[k]=v
    

def get_sorted_indices(lst):
    # Step 1: Pair each element with its index
    indexed_list = [(value, index) for index, value in enumerate(lst)]
    # Step 2: Sort the list of tuples based on the first element (the value)
    sorted_indexed_list = sorted(indexed_list, key=lambda x: x[0], reverse=True)
    # Step 3: Extract the indices from the sorted list
    sorted_indices = [index for value, index in sorted_indexed_list]
    return sorted_indices



# load og the dictionary with all information for each occupation
with open('/dizionario_ordinato_occupazioni.json', 'r', encoding ='utf-8') as file:
    dd = json.load(file)
for i,v in dd.items():
    dd[i]['occupation_alternative_label_vet'] = dd[i]['occupation_alternative_label'].split('\n')
    dd[i]['occupation_alternative_label_vet'].append(dd[i]['ISCO_label'])
    dd[i]['occupation_alternative_label_vet'].append(dd[i]['occupation_label'])
    dd[i]['occupation_alternative_label_vet'] = list(set(dd[i]['occupation_alternative_label_vet']))
    

# function to find perfect match in the title
with open('/diz_corr.json', 'r', encoding='utf-8') as file:
    diz_corrispondenze=json.load(file)
mat_corr = {}
for id, corr in diz_corrispondenze.items():
    for c in corr:
        if id not in list(mat_corr.keys()):
            mat_corr[id]=[c]
        else:
            mat_corr[id].append(c)  




# functions which performs key and vectorial search on weaviate and retrieved the mostre related occupation
# append the perfect match title, if any
# group occupations by 4-digit ISCO group and then built the short descriptions for each job posting
def materials(k, alpha, num_vett, num_text):
    v = data_text_diz[k]
    dd_text = clean_job_description(v).replace('/', 'and')
    dd_listati = clean_job_description(listati_diz[k])
    # search in weavate database
    retrived= wea_complete_occ(dd_listati,dd_text, alpha, num_vett, num_text, x, y)
    total=list(retrived[-1].values())
    list_mat1 = [l.strip() for l in total if l.strip()]
    codes1=[i for i in list(retrived[-1].keys())]
    
    # building of the couple query, occupation for reranking
    list_mat= [[dd_text,t.strip()] for t in list_mat1]
    
    # rerancking and extraction of the first 10 occupations
    score = reranker.compute_score(list_mat)
    sorted_pos=get_sorted_indices(score)[0:10]
    
    # we append eventually occupations with perfect matches in the title
    if k in list(mat_corr.keys()):
        codes2 = [l for l in mat_corr[k]]  
    else:
        codes2=[]
    codes = [codes1[i] for i in sorted_pos]
    codes += codes2
    codes = list(set(codes))
    
    # we build the short descriptions by groupping by 4-digit ISCO category
    testo_RAG = {}
    for i in codes:
        ISCO_codes=dd[i]['ISCO_code']
        ISCO_desc=dd[i]['ISCO_description']
        ISCO_label=dd[i]['ISCO_label']
        if ISCO_codes in testo_RAG.keys():
            testo_RAG[ISCO_codes] = testo_RAG[ISCO_codes] + "\n\nOccupation Title: " + dd[i]['occupation_label']  + "\nOccupation Description: " + dd[i]["occupation_description"] #+ "\nOccupation Alternative Titles: " + item["occup_alt_label"].replace("\n", ", ")
        else:
            testo_RAG[ISCO_codes] = "\n\n----------------------------------------------------------------\n----------------------------------------------------------------\nISCO Category: " + ISCO_label + "\nISCO category code: " + ISCO_codes+ "\nCategory Description: " + ISCO_desc
            testo_RAG[ISCO_codes] = testo_RAG[ISCO_codes] + "\nOccupations in this category that might be related to the job ad:\n"
            testo_RAG[ISCO_codes] = testo_RAG[ISCO_codes] + "\nOccupation Title: " + dd[i]["occupation_label"]  + "\nOccupation Description: " + dd[i]["occupation_description"] #+ "\nOccupation Alternative Titles: " + item["occup_alt_label"].replace("\n", ", ")

        codes_finali = []
        text_return = ""
    for key in testo_RAG.keys():
        text_return+=testo_RAG[key]
    return list(set([codes1[i][0:4] for i in sorted_pos])), text_return


# collection of all IDs of the job postings
id_list=list(data_text_diz.keys())


# loop over each job posting IDs and saving of the short descriptions crated by "materials(k, alpha, num_vett, num_text)"
diz_emb={'id':[],'codici':[], 'testo':[]}
diz_temp=[]
controll = False
for idx in tqdm(range(len(id_list))):
    diz_emb['id'].append(id_list[idx])
    mat = materials(id_list[idx], [0.1,0.5,1.0], [5,5,10], 10)
    diz_emb['codici'].append(mat[0])
    diz_emb['testo'].append(mat[1].strip())
    diz_temp.append({'id':id_list[idx],'codici':mat[0], 'testo':mat[1].strip()})
    if idx%10==0:
        if idx%1000==0:
            nome = f'/home/a.alessandrelli/LLM/EUROPA/v10/embedding_{idx}.jsonl'
            controll=True
        if controll:
            with open(nome, 'w', encoding='utf-8') as file:
                for d in diz_temp:
                    file.write(json.dumps(d, ensure_ascii=False) + '\n')
            controll = False
        else:
            with open(nome, 'a', encoding='utf-8') as file:
                for d in diz_temp:
                    file.write(json.dumps(d, ensure_ascii=False) + '\n')
        diz_temp=[]
with open(nome, 'a', encoding='utf-8') as file:
                for d in diz_temp:
                    file.write(json.dumps(d, ensure_ascii=False) + '\n')
    
    
df_emb = pd.DataFrame(diz_emb)

df_emb.to_csv('/dati_embedding.csv', index=False)