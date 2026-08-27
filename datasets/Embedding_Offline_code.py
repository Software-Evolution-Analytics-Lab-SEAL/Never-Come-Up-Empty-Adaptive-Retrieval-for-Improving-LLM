# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# ## Improve the performance by embedding the questions and answers (sentence by sentence) offline

# %%
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm
import torch
import json
from tqdm import tqdm
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import nltk
from nltk.tokenize import sent_tokenize

# Download the Punkt Tokenizer Models (only need to do this once)
nltk.download('punkt')


# %%
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
print("Denpendency !")

# os.chdir('path')
# %%
import json
#     eval_data = json.load(f)
with open('stack_overflow_python_java.json', 'r') as f:
    combined_data = json.load(f)
print("file loaded")

# %%
# combined_data


# %%
# Function to filter the data
def filter_data(data):
    filtered_data = []
    for key, value in data.items():
        # Check if 'accepted_answer' exists
        if 'accepted_answer' in value:
            # Remove 'accepted_answer_id' attribute
            value.pop('accepted_answer_id', None)
            filtered_data.append(value)
    return filtered_data


# %%
cleaned_combined_data = filter_data(combined_data)


# %%
# len(cleaned_combined_data),len(combined_data)


# %%
tag_list = [item['tags'] for item in cleaned_combined_data]


# %%
import re
from nltk.tokenize import sent_tokenize

def clean_sentence(sen):
    sen = sen.replace('\n', ' ')

    # Remove all tags except <p> and </p>
    sen = re.sub(r'(?i)<(?!\/?p\b)[^>]*>', '', sen)

    sen = re.sub(r'\s+', ' ', sen).strip()
    return sen

def split_and_clean_p_tags(sentences):
    new_sentences = []
    for sen in sentences:
        # Remove <p> and </p> tags
        cleaned_sentence = re.sub(r'</?p>', '', sen)
        cleaned_sentence = cleaned_sentence.strip()  # Remove leading/trailing whitespace

        if cleaned_sentence:  # Check if the sentence is not empty
            new_sentences.append(cleaned_sentence)

    return new_sentences
def custom_sent_tokenize(text):
    text = text.replace('&quot;', '"')

    sections = re.findall(r'(?:<pre><code>.*?</code></pre>|<li><p>.*?</li>|<ol><p>.*?</ol>|<ul><p>.*?</ul>|<menu><p>.*?</menu>|<p>.*?</p>)', text, flags=re.DOTALL)

    output_sentences = []
    code_snippets = []
    urls = []
    url_positions = []
    code_positions = []
    # print(sections)
    position_counter = 0
    for section in sections:
        if '<pre><code>' in section:
            match = re.search(r'<pre><code>(.*?)</code></pre>', section, flags=re.DOTALL)
            if match:
                code = match.group(1)
                code_snippets.append(code)
                code_positions.append(position_counter)
            else:
                code = None 
            # position_counter += 1
        elif '<li>' in section:
            list_items = re.findall(r'<li>(.*?)</li>', section, flags=re.DOTALL)
            list_sentences = [clean_sentence(item) for item in list_items]
            output_sentences.extend(list_sentences)
            extracted_urls = re.findall(r'<a href="(.*?)"', section)
            
            if len(extracted_urls)>0:
                for u in extracted_urls:
                    urls.append(u)
                    url_positions.append(position_counter)  
            if len(list_sentences)>0:
                position_counter += 1
        elif '<ul>' in section:  
            list_items = re.findall(r'<ul>(.*?)</ul>', section, flags=re.DOTALL)
            list_sentences = [clean_sentence(item) for item in list_items]
            output_sentences.extend(list_sentences)
            extracted_urls = re.findall(r'<a href="(.*?)"', section)
            if len(extracted_urls)>0:
                for u in extracted_urls:
                    urls.append(u)
                    url_positions.append(position_counter)
            
            if len(list_sentences)>0:
                position_counter += 1
        
        elif '<ol>' in section:  
            list_items = re.findall(r'<ol>(.*?)</ol>', section, flags=re.DOTALL)
            list_sentences = [clean_sentence(item) for item in list_items]
            output_sentences.extend(list_sentences)
            extracted_urls = re.findall(r'<a href="(.*?)"', section)
            if len(extracted_urls)>0:
                for u in extracted_urls:
                    urls.append(u)
                    url_positions.append(position_counter)
            
            if len(list_sentences)>0:
                position_counter += 1  

        elif '<menu>' in section:  
            list_items = re.findall(r'<menu>(.*?)</menu>', section, flags=re.DOTALL)
            list_sentences = [clean_sentence(item) for item in list_items]
            output_sentences.extend(list_sentences)
            extracted_urls = re.findall(r'<a href="(.*?)"', section)
            
            if len(extracted_urls)>0:
                for u in extracted_urls:
                    urls.append(u)
                    url_positions.append(position_counter)
            
            if len(list_sentences)>0:
                position_counter += 1 

        elif '<p>' in section:
            paragraph = re.search(r'<p>(.*?)</p>', section, flags=re.DOTALL).group(1)
            sentences = sent_tokenize(paragraph)
            cleaned_sentences = [clean_sentence(sen) for sen in sentences]
            output_sentences.extend(cleaned_sentences)

            for cs in range(len(cleaned_sentences)):
                # extracted_urls = re.findall(r'<a href="(.*?)".*?>', cs)
                extracted_urls = re.findall(r'<a href="(.*?)"', sentences[cs], flags=re.DOTALL)
                if len(extracted_urls)>0:
                    for u in extracted_urls:
                        urls.append(u)
                        url_positions.append(position_counter)
                        
                position_counter += 1
           
    final_output = split_and_clean_p_tags(output_sentences)

    knowledge_dictionary = []
    for i in range(len(output_sentences)):
        code_loc = []
        urls_loc=[]
        for c in range(len(code_positions)):
            if code_positions[c]>i:
                code_loc.append(c)
        for u in range(len(url_positions)):
            if url_positions[u]==i:
                urls_loc.append(u)
        knowledge_tuple = (code_loc,urls_loc)
        knowledge_dictionary.append(knowledge_tuple)
    # knowledge_dictionary stores the information to track the related code snippets and urls for 
    # each sentence in corresponding position. The 1st postition of tuple stores the indexes of code snippets
    # the second position of tuple stores the indexes of urls
    return final_output, code_snippets, urls, knowledge_dictionary


# %%
# print(cleaned_combined_data[72]["accepted_answer"])


# %%
# print(cleaned_combined_data[1000]["accepted_answer"])


# %%
# cleaned_combined_data[72]["question"]


# %%
# testing_str = '<p>I\'ve just put together what you may be looking for: <a href="http://www.graphdracula.net" rel="noreferrer">http://www.graphdracula.net</a></p>'
# test = re.findall(r'<a href="(.*?)"', testing_str)
# print(test)


# %%
# cleaned_text = cleaned_combined_data[72]["accepted_answer"]
# # cleaned_text = preprocess_text_sentence(cleaned_combined_data[72]["accepted_answer"])
# answer_sentences, code, urls, map= custom_sent_tokenize(cleaned_text)


# %%
# # print(answer_sentences)
# for i in range(len(answer_sentences)): 
#     print(str(i)+":"+answer_sentences[i])
#     print(map[i])
#     print('-------------------')


# %%
# print(answer_sentences[10])


# %%
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

device = "cuda"

para_tokenizer = AutoTokenizer.from_pretrained("humarin/chatgpt_paraphraser_on_T5_base")

para_model = AutoModelForSeq2SeqLM.from_pretrained("humarin/chatgpt_paraphraser_on_T5_base").to(device)

def paraphrase(
    question,
    num_beams=5,
    num_beam_groups=5,
    num_return_sequences=5,
    repetition_penalty=10.0,
    diversity_penalty=3.0,
    no_repeat_ngram_size=2,
    temperature=0.7,
    max_length=128
):
    input_ids = para_tokenizer(
        f'paraphrase: {question}',
        return_tensors="pt", padding="longest",
        max_length=max_length,
        truncation=True,
    ).input_ids
    
    input_ids = input_ids.to(device)

    outputs = para_model.generate(
        input_ids, temperature=temperature, repetition_penalty=repetition_penalty,
        num_return_sequences=num_return_sequences, no_repeat_ngram_size=no_repeat_ngram_size,
        num_beams=num_beams, num_beam_groups=num_beam_groups,
        max_length=max_length, diversity_penalty=diversity_penalty
    )

    res = para_tokenizer.batch_decode(outputs, skip_special_tokens=True)

    return res
print("T5 !")

# %%
model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
print("Model Loaded !")

# %%
embeddings_data = []

for data_item in tqdm(cleaned_combined_data):
    knowledge_pockets = {}
    raw_question = data_item['question']
    raw_accepted_answer = data_item['accepted_answer']

    # Compute embedding for the question
    question_embedding = model.encode(raw_question, convert_to_tensor=True)

    # Split the accepted answer into sentences
    answer_sentences, code_snippets, urls, knowledge_dictionary= custom_sent_tokenize(raw_accepted_answer)
    # cleaned_text = preprocess_text_sentence(raw_accepted_answer)
    # answer_sentences, code_snippets= custom_sent_tokenize(make_human_readable(cleaned_text))
    
    # Compute embeddings for each sentence in the answer
    # answer_embeddings = [model.encode(sentence, convert_to_tensor=True) for sentence in answer_sentences]
    knowledge_pockets['code_snippets'] = code_snippets
    knowledge_pockets['urls'] = urls
    knowledge_pockets['knowledge_dictionary'] = knowledge_dictionary
    # Store original text and embeddings in a single dictionary
    embeddings_data.append({
        'raw_question': raw_question,
        'question_embedding': question_embedding,
        'raw_accepted_answer': raw_accepted_answer,
        # 'answer_embeddings': answer_embeddings,
        'knowledge':knowledge_pockets
    })
print("Processing Done !")

# %%
# len(embeddings_data[0]['answer_embeddings'])


# %%
import pickle
with open('CODE_POST_OVERALL-EMBEDDINGS_DATA.pkl', 'wb') as f:
    pickle.dump(embeddings_data, f)

print("V1 Done !")
# %%
with open('CODE_POST_OVERALL-EMBEDDINGS_DATA.pkl', 'rb') as f:
    OVO_data = pickle.load(f)


# %%
# tag_list[1]


# %%
# if len(OVO_data) == len(tag_list):
#     for data_item, tag in zip(OVO_data, tag_list):
#         data_item['tag'] = tag
# else:
#     print("The lengths of OVO_data and tags_list do not match!")

# with open('OVERALL-EMBEDDINGS_DATA_V2.pkl', 'wb') as f:
#     pickle.dump(OVO_data, f)


# %%
# OVO_data[0]


# %%
for single_item in tqdm(OVO_data):
    raw_accepted_answer = single_item['raw_accepted_answer']
    
    # Preprocess the accepted answer and tokenize it into sentences
    answer_sentences, code_snippets, urls, knowledge_dictionary= custom_sent_tokenize(raw_accepted_answer)
    
    # cleaned_text = preprocess_text_sentence(raw_accepted_answer)
    # answer_sentences, code= custom_sent_tokenize(make_human_readable(cleaned_text))
    
    # Add the computed answer_sentences to the dictionary
    single_item['answer_sentences'] = answer_sentences

with open('CODE_POST_OVERALL-EMBEDDINGS_DATA_V2.pkl', 'wb') as f:
    pickle.dump(OVO_data, f)
print("V1 Done !")

# %%
# with open('CODE_POST_OVERALL-EMBEDDINGS_DATA_V2.pkl', 'rb') as f:
#     OVO_data_V2 = pickle.load(f)


# %%
# OVO_data_V2[0]


# %%



