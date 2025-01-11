import os
import re
import json
import numpy as np 
import fitz
import pandas as pd
import faiss

import torch
from transformers.utils import is_flash_attn_2_available 
from transformers import BitsAndBytesConfig,AutoModelForCausalLM, AutoTokenizer

import nltk
from nltk.tokenize import word_tokenize

from sentence_transformers import SentenceTransformer,util

from tqdm.auto import tqdm

class PDF_analyze():
    def __init__(self,directory,json_file):
        self.directory=directory
        self.json_file=json_file
        pass

    def read_json(self):
        '''
        Считываем значения с json-файла
        '''
        with open(self.json_file, 'r') as file:
            data = json.load(file)
        return data
    
    def pdf_dir(self,):
        ''' 
        Получаем директорий с пдф файлами
        '''
        current_dir_os = os.getcwd() 
        pdf_dir=current_dir_os+'\\'+self.directory # можно свое название папки использовать
        print (pdf_dir)
        return pdf_dir

    def filepath_list(self,pdf_dir):
        '''
        Создаем список с путем до каждого файла
        '''
        files_path=[]
        files = os.listdir(pdf_dir)
        for i in files:
            file_path=pdf_dir+'\\'+i
            files_path.append(file_path)
        print ("[INFO]: 101 - read directory")
        return files_path

    def pdf_filter(self,text:str):
        '''
        настраеваемый фильтр текста 
        '''
        # надо дополнить , а то хуйня
        clean_text=text.replace('\n',' ').strip()
        return clean_text

    def read_file(self,path,main_start,mail_finish):
        '''
        переводим .pdf в формат для дальнейшего анализа
        '''
        doc=fitz.open(path) 
        pageXtext=[]
        for pg_number,page in tqdm(enumerate(doc)):
            page_text=page.get_text() # читаем тест со страницы
            #  фильтруем
            cleaned_text=self.pdf_filter(page_text)
            #  формируем список в соотвествии с требования 
            pageXtext.append({
                            'file_name': os.path.basename(path),
                            'page_number':pg_number-main_start, 
                            'page_char_count':len(cleaned_text),
                            'page_word_count':len(cleaned_text.split(' ')),
                            'page_sentence_count':len(cleaned_text.split('. ')),
                            'page_tokens_count': len(cleaned_text)/4,
                            'text':cleaned_text,
                                }) 
        print (f"[INFO]: 102 - processed file {os.path.basename(path)}")
        return pageXtext
    
    def processed_pdf(self,files_list):
        '''
        Проходим обработкой по всему тексту
        '''
        text_bookXpage=[]
        book_info=self.read_json()
        # data['books'][0]['file_name']
        for filepath in files_list:
            #  defining the beginning and end of an informative text
            for book in book_info["books"]:
                if book["file_name"] == os.path.basename(filepath):
                    main_start = book["main_start"]
                    main_finish = book["main_finish"]
                    print ('[INFO]: found book')

            proc_text=self.read_file(filepath,main_start,main_finish)
            text_bookXpage.extend(proc_text)
    
        print (f"[INFO]: 103 - processed file's directory")
        return text_bookXpage

# # Работает 
# PDF=PDF_analyze('training\qa_nn\pdf','training\qa_nn\help_info.json')
# path=PDF.filepath_list(PDF.pdf_dir())
# print(path)
# # page_and_text=PDF.read_file(path[0])  
# # print(page_and_text)

# books_text=PDF.processed_pdf(path)

class Text_preprocessing():
    def __init__(self,text_boolXpage,chunk_size):
        self.text_boolXpage=text_boolXpage
        self.chunk_size=chunk_size
        pass

    def text2sentence(self,):
        '''
        Выделяем новые элементы (предложения)
        '''
        for item in tqdm(self.text_boolXpage):
            item['sentence'] = nltk.sent_tokenize(item['text'])  
            item['sentence']=[str(sentence) for sentence in item['sentence']]
            item['sentence_count']=len(item['sentence'])
            del item['page_sentence_count']
        print("[INFO]: 201 - split into sentences")

    def list_chunk(self,item):
        '''
        Разбиваем элемент books['sentence'] на более мелкие части (по 7 предложений)
        '''
        # сюда передаем список с предложениями и делим
        return [item[i:i+self.chunk_size] for i in range(0,len(item),self.chunk_size)]
    
    def sentence2chunk(self,):
        # разбиваем на опр кол-во предложений и составляем новые item
        for item in tqdm(self.text_boolXpage):
            item['sentence_chunk']=self.list_chunk(item['sentence'])
        print ("[INFO]: 202 - split into chunk")

    def chunking(self):
        '''
        делаем части предложений как новый элемент для создания embeddings
        '''
        pageXchunk=[]
        for item in tqdm(self.text_boolXpage):
            for sentence in item['sentence_chunk']:
                chunk_dict={}
                chunk_dict['file_name']=item['file_name']
                chunk_dict['page_number']=item['page_number']
                # филтруем мб (тут пока так и есть)
                join_sent_chunk=''.join(sentence).replace('  ',' ').strip()
                join_sent_chunk=re.sub(r"\.([A-Z])",r". \1",join_sent_chunk  )

                chunk_dict['sentence_chunk']=join_sent_chunk
                chunk_dict['word_count']=len(word_tokenize(join_sent_chunk))
                chunk_dict['token_count']=len(join_sent_chunk)/4

                pageXchunk.append(chunk_dict)
        print ("[INFO]: 203 - finish chunking")
        return pageXchunk
    
    
# TP=Text_preprocessing(books_text,7)
# TP.text2sentence()
# TP.sentence2chunk()
# pa=TP.chunking()
# # # print (pa[0])
# # # print(books_text[5])


class Text_filter():

    @staticmethod
    def token_filter(data,min_token_length):
        data=pd.DataFrame(data)
        return data[data['token_count']>min_token_length].to_dict(orient='records')


class Make_embedding():
    '''
    Создает тензор эмбеддингов
    '''
    def __init__(self,voc_text,model_name):
        self.embd_model=SentenceTransformer(model_name_or_path=model_name,device='cuda')
        self.voc_text=voc_text

    def model(self):
        return self.embd_model

    def encode_item(self,item):
        '''Переводим DataFrame - List'''
        chunked_embd=[txt['sentence_chunk'] for txt in item]
        # print (len(chunked_embd))
        return chunked_embd

    def embedding_process(self):
        chunk_embedings=self.embd_model.encode(self.encode_item(self.voc_text),
                                       batch_size=16,
                                       convert_to_tensor=True)
        print ("[INFO]: 301 - Make Embedding")
        return chunk_embedings
        
class Vector_base():
    '''Create vectore base '''
    def __init__(self,embedding,nlist):
        self.dataframe=pd.DataFrame(embedding.cpu().numpy())
        self.nlist=nlist

    def base_formation(self):
        quantizer = faiss.IndexFlatL2(self.dataframe.shape[1])
        index = faiss.IndexIVFFlat(quantizer, self.dataframe.shape[1], self.nlist)
        index.train(self.dataframe)
        index.is_trained
        index.add(self.dataframe)
        print('[INFO]: 401 - Database is ready ')
        return index
    
    def save_vector_base(self,vector_base,name):
        faiss.write_index(vector_base, name) 
        print('[INFO]: 402 - Database is saved')

    # @property
    # def idx():
    #     return self.index

    def search_bd(self,embd_model,database_name,query_txt,k_near,nprobe):
        index= faiss.read_index(database_name)
        query = embd_model.encode([query_txt])
        index.nprobe = nprobe
        Dist, Idx = index.search(query, k_near)  # search
        print('[INFO]: 403 - Database is loaded')

        return Dist,  Idx

# Emb=Make_embedding(pa,'sentence-transformers/all-MiniLM-L12-v2')
# Emb.processed()

class Response_system():
    def __init__(self,texts,similar_idx):
        self.texts=texts
        self.similar_idx=similar_idx
        
    def similar_texts(self,texts,idxs):
        similar_texts = (texts[i]["sentence_chunk"] for i in idxs[0])
        return similar_texts 

    def _quantization_config(self):

        quantization = BitsAndBytesConfig(load_in_4bit=True,
                                                bnb_4bit_compute_dtype=torch.float16)
        # Setup Flash Attention 2 for faster inference, default to "sdpa" or "scaled dot product attention" if it's not available
        if (is_flash_attn_2_available()) and (torch.cuda.get_device_capability(0)[0] >= 8):
            attn_implementation = "flash_attention_2"
        else:
            attn_implementation = "sdpa" # scale dot production
        print(f"[INFO] Using attention implementation: {attn_implementation}")  

        return quantization,attn_implementation     
    
    def llm_model(self,model_name):

        quant_conf,attn_implementation=self._quantization_config()
        # начинаем с претрена все тренировочки и токенизатор
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="cuda",
            attn_implementation=attn_implementation,
            quantization_config=quant_conf
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name)     

        return model ,tokenizer   
    
    def make_response(self,promt,tokenizer,model):
        
        combined_text = f"Promt: {promt}\nSimilarity text:\n" + "\n".join(self.similar_texts(self.texts,self.similar_idx))
        # inputs = tokenizer(combined_text, return_tensors='pt', padding=True, truncation=True)
        # Создайте шаблон приглашения для модели, настроенной на основе инструкций
        messages = [
            {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
            {"role": "user", "content": combined_text}
                    ]
        # Примените шаблон чата
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False, # keep as raw text (not tokenized)
            add_generation_prompt=True
        )
        # Наш следующий шаг - выделить этот форматированный текст и передать его методу generate() нашей модели.
        # Мы позаботимся о том, чтобы наш токенизированный текст был на том же устройстве, что и наша модель (GPU), используя "to("cuda")".
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

        # Генерировать выходные данные, передаваемые по токенизированному входу
        generated_ids = model.generate(
                                        **model_inputs,
                                        max_new_tokens=128
                                            )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return response
    
    


        
