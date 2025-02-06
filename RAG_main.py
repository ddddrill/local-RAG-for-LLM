import os
import re
import json
import numpy as np 
import fitz
import pandas as pd
import faiss
import time

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


    @staticmethod
    def search_bd(embd_model,database_name,query_txt,k_near,nprobe):
        # ВОТ ТУТ НАДО И ВЬЕБАТЬ КЕШ
        index= faiss.read_index(database_name)
        query = embd_model.encode([query_txt])
        index.nprobe = nprobe
        Dist, Idx = index.search(query, k_near)  # search
        print('[INFO]: 403 - Database is loaded')

        return Dist,  Idx
    

class CacheCore:
    @staticmethod
    def cache_init(embedding_dim):
        '''Основа для дальнейшей базы данных кеша'''
        index = faiss.IndexFlatL2(embedding_dim)  # Пока оставим эту длину
        if index.is_trained:
            print("Index trained")
        return index

    @staticmethod
    def retrieve_cache(json_file):
        '''Считываем данные кеша или создаем с нуля'''
        try:
            with open(json_file, "r") as file:
                cache = json.load(file)
        except FileNotFoundError:
            cache = {"questions": [], "embeddings": [], "answers": [], "response_text": []}
        return cache

    @staticmethod
    def store_cache(json_file, cache):
        '''Сохраняет файл, содержащий данные кэша, на диск.'''
        def convert(o):
            if isinstance(o, np.generic):
                return o.item()  # Конвертируем numpy-типы в Python-типы
            raise TypeError(f"Несериализуемый тип: {type(o)}")

        # Рекурсивно обрабатываем всю структуру кэша
        def recursive_convert(obj):
            if isinstance(obj, dict):
                return {k: recursive_convert(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [recursive_convert(v) for v in obj]
            else:
                return convert(obj) if isinstance(obj, np.generic) else obj

        processed_cache = recursive_convert(cache)
        with open(json_file, "w") as file:
            json.dump(processed_cache, file, default=convert)

class Semantic_Cache(CacheCore):
    def __init__(self,texts ,database,cache_file, embd_model, threshold, max_response, eviction_policy):
        self.texts=texts
        self.database=database
        self.cache_file = cache_file
        self.embd_model = embd_model
        # Параметры фильтрации
        self.threshold = threshold
        self.max_response = max_response
        self.eviction_policy = eviction_policy

        # Инициализация индекса Faiss, модели и загрузка кеша
        embedding_dim = self.embd_model.get_sentence_embedding_dimension()
        self.index= CacheCore.cache_init(embedding_dim)
        self.cache = CacheCore.retrieve_cache(self.cache_file)


    def evict(self):
        """Удаляет элемент из кэша в соответствии с политикой удаления."""
        if self.eviction_policy and len(self.cache["questions"]) > self.max_response:
            for _ in range((len(self.cache["questions"]) - self.max_response)):
                if self.eviction_policy == "FIFO":
                    self.cache["questions"].pop(0)
                    self.cache["embeddings"].pop(0)
                    self.cache["answers"].pop(0)
                    self.cache["response_text"].pop(0)

    
    def request_cache(self, query):
        start_time = time.time()
        try:
            # Получаем эмбеддинги для запроса
            embedding_cache = self.embd_model.encode([query])
            # Ищем ближайшего соседа в индексе
            self.index.nprobe = 10
            Dist_cache, Idx_cache = self.index.search(embedding_cache, 1)
            print (Dist_cache)
            # Если ответ найден в кеше и расстояние меньше порога
            if Dist_cache[0] >= 0:  # Проверяем наличие ответов
                if Idx_cache[0][0] >= 0 and Dist_cache[0][0] <= self.threshold:  # Проверяем порог
                    row_id = int(Idx_cache[0][0])

                    print("Answer recovered from Cache.")
                    print(f"Crossed the threshold")
                    print(f"Found cache in row: {row_id} with score {Dist_cache[0][0]:.3f}")
                    print(f"response_text: " + self.cache["response_text"][row_id])

                    end_time = time.time()
                    elapsed_time = end_time - start_time
                    print(f"Time taken: {elapsed_time:.3f} seconds")
                    return Idx_cache

            # Если ответа нет в кеше, обращаемся к базе данных
            Dist_base, Idx_base=Vector_base.search_bd(self.embd_model,self.database,query,4,10)

            # Обновляем кеш
            
            self.cache["questions"].append(query)
            self.cache["embeddings"].append(embedding_cache[0].tolist())
            self.cache["answers"].append(Idx_base[0][0])  # Сохраняем ответ
            self.cache["response_text"].append(self.texts[Idx_base[0][0]]['sentence_chunk'])

            print("Answer recovered from main Database.")
            print(f"response_text: {self.texts[Idx_base[0][0]]['sentence_chunk']}")

            DF=pd.DataFrame(embedding_cache)
            # Добавляем эмбеддинг в индекс
            self.index.add(DF)
            # Удаляем старые записи, если необходимо
            self.evict()

            # Сохраняем кеш на диск
            CacheCore.store_cache(self.cache_file, self.cache)

            end_time = time.time()
            elapsed_time = end_time - start_time
            print(f"Time taken: {elapsed_time:.3f} seconds")

            return Idx_base
        except Exception as e:
            raise RuntimeError(f"Error during 'request_cache' method: {e}")


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
            {"role": "system", "content": "How can I help you ?"},
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
    
    


        
