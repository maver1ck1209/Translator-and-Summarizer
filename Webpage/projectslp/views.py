from django.shortcuts import render, redirect
import PyPDF2
import tensorflow as tf
import numpy as np
from .transformerfolder.abstransformer import Transformer, masks
import pickle
from transformers import MBartForConditionalGeneration,MBart50TokenizerFast
import nltk
from nltk.corpus import stopwords
from nltk.cluster.util import cosine_distance
from nltk.tokenize import sent_tokenize
import numpy as np
import networkx as nx
import re
nltk.download('stopwords')
nltk.download('punkt')
from transformers import pipeline #abstractive summarization
from summarizer.sbert import SBertSummarizer

################################################### MISC #################################################################################

#abstractive summarization
pickle_in=open(r"C:\Users\maver\OneDrive\Documents\Giridhar\sem6\SLP -CSE3119\JCOMP\Summary_translator\Webpage\projectslp\transformerfolder\document.pkl","rb")
document=pickle.load(pickle_in)
pickle_in.close()
pickle_in=open(r"C:\Users\maver\OneDrive\Documents\Giridhar\sem6\SLP -CSE3119\JCOMP\Summary_translator\Webpage\projectslp\transformerfolder\summary.pkl","rb")
summary=pickle.load(pickle_in)
pickle_in.close()
# for decoder sequence
summary = summary.apply(lambda x: '<go> ' + x + ' <stop>')
filters = '!"#$%&()*+,-./:;=?@[\\]^_`{|}~\t\n'
oov_token = '<unk>'
document_tokenizer = tf.keras.preprocessing.text.Tokenizer(oov_token=oov_token)
summary_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters=filters, oov_token=oov_token)
document_tokenizer.fit_on_texts(document)
summary_tokenizer.fit_on_texts(summary)
# hyper-params
num_layers = 4
d_model = 128
dff = 512
num_heads = 8
EPOCHS = 23
transformer = Transformer(
    num_layers, 
    d_model, 
    num_heads, 
    dff,
    len(document_tokenizer.word_index) + 1, 
    len(summary_tokenizer.word_index) + 1, 
    pe_input=len(document_tokenizer.word_index) + 1, 
    pe_target=len(summary_tokenizer.word_index) + 1,
)
checkpoint_path = "checkpoints"

# Load checkpoints
ckpt = tf.train.Checkpoint(transformer=transformer)
ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)
if ckpt_manager.latest_checkpoint:
    ckpt.restore(ckpt_manager.latest_checkpoint)
    print ('Latest checkpoint restored!!')

def evaluate(input_document):
    input_document = document_tokenizer.texts_to_sequences([input_document])
    input_document = tf.keras.preprocessing.sequence.pad_sequences(input_document, maxlen=400, padding='post', truncating='post')

    encoder_input = tf.expand_dims(input_document[0], 0)

    decoder_input = [summary_tokenizer.word_index["<go>"]]
    output = tf.expand_dims(decoder_input, 0)
    for i in range(75):
        enc_padding_mask, combined_mask, dec_padding_mask = masks.create_masks(encoder_input, output)

        predictions, attention_weights = transformer(
            encoder_input, 
            output,
            False,
            enc_padding_mask,
            combined_mask,
            dec_padding_mask
        )

        predictions = predictions[: ,-1:, :]
        predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)

        if predicted_id == summary_tokenizer.word_index["<stop>"]:
            return tf.squeeze(output, axis=0), attention_weights

        output = tf.concat([output, predicted_id], axis=-1)

    return tf.squeeze(output, axis=0), attention_weights

def summarize(input_document):
    # not considering attention weights for now, can be used to plot attention heatmaps in the future
    summarized = evaluate(input_document=input_document)[0].numpy()
    summarized = np.expand_dims(summarized[1:], 0)  # not printing <go> token
    return summary_tokenizer.sequences_to_texts(summarized)[0]  # since there is just one translated document

#extractive summarization

def read_article(text):        
    sentences =[]        
    sentences = sent_tokenize(text)    
    for sentence in sentences:        
        sentence.replace("[^a-zA-Z0-9]"," ")     
    return sentences

def sentence_similarity(sent1,sent2,stopwords=None):
    if stopwords is None:
        stopwords = []
    sent1 = [w.lower() for w in sent1]
    sent2 = [w.lower() for w in sent2]
    all_words = list(set(sent1 + sent2))
    vector1 = [0] * len(all_words)
    vector2 = [0] * len(all_words)
    #build the vector for the first sentence
    for w in sent1:
        if not w in stopwords:
            vector1[all_words.index(w)]+=1    
    #build the vector for the second sentence
    for w in sent2:
        if not w in stopwords:
            vector2[all_words.index(w)]+=1
    return 1-cosine_distance(vector1,vector2)

def build_similarity_matrix(sentences,stop_words):
    #create an empty similarity matrix
    similarity_matrix = np.zeros((len(sentences),len(sentences)))
    for idx1 in range(len(sentences)):
        for idx2 in range(len(sentences)):
            if idx1!=idx2:
                similarity_matrix[idx1][idx2] = sentence_similarity(sentences[idx1],sentences[idx2],stop_words)
    return similarity_matrix

def generate_summary(text,top_n):
    stop_words = stopwords.words('english')
    summarize_text = []
    # Step1: read text and tokenize
    sentences = read_article(text)
    # Steo2: generate similarity matrix across sentences
    sentence_similarity_matrix = build_similarity_matrix(sentences,stop_words)
    # Step3: Rank sentences in similarirty matrix
    sentence_similarity_graph = nx.from_numpy_array(sentence_similarity_matrix)
    scores = nx.pagerank(sentence_similarity_graph)
    #Step4: sort the rank and place top sentences
    ranked_sentences = sorted(((scores[i],s) for i,s in enumerate(sentences)),reverse=True)
    # Step 5: get the top n number of sentences based on rank    
    for i in range(top_n):
        summarize_text.append(ranked_sentences[i][1])
    # Step 6 : outpur the summarized version
    return " ".join(summarize_text)

def abssummarizer(str): #abstractive summarization using hugging face
  summarizer = pipeline("summarization")
  text = str
  summary = summarizer(text, max_length=100, min_length=30, do_sample=False)
  return summary
 ######################################################## VIEWS ##########################################################################

# Create your views here.
def index(request):
    return render(request,"index.html")

def summary_page(request):
    if request.method == 'POST':
        documents = request.POST['input']
        t = request.POST['sumtype']
        nol = request.POST['nol']
        if t == 'a':
            doc = summarize(documents)
        elif t == 'e':
            doc = generate_summary(documents, int(nol))
        context = {
            'title': 'Summarized Output',
            'page_content': doc,
        }
        return render(request, 'pdf_document.html', context)
    else:
        return render(request,'summary.html')

def translater(request):
    global doc
    languages = {'malayalam':'ml_IN', 'tamil':'ta_IN', 'hindi':'hi_IN', 'french':'fr_XX', 'japanese':'ja_XX' }
    if request.method == 'POST':
        lang = request.POST['lang']
        model=MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
        tokenizer=MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50-many-to-many-mmt",src_lang=languages[lang])
        if request.POST['option']== 'upload':
            if request.FILES['pdf_document']:
                pdf_file = request.FILES['pdf_document']
                pdf_reader = PyPDF2.PdfReader(pdf_file)
                num_pages = len(pdf_reader.pages)
                page_content = []
                for page_num in range(num_pages):
                    page = pdf_reader.pages[page_num]
                    page_text = page.extract_text()
                    page_content.append(page_text)
                documents = ''
                for i in page_content:
                    documents += i
                model_inputs=tokenizer(documents,return_tensors='pt')
                gtokens=model.generate(**model_inputs,forced_bos_token_id=tokenizer.lang_code_to_id["en_XX"])
                doc=tokenizer.batch_decode(gtokens,skip_special_tokens=True)
        elif request.POST['option']== 'text':
            documents = request.POST['text']
            model_inputs=tokenizer(documents,return_tensors='pt')
            gtokens=model.generate(**model_inputs,forced_bos_token_id=tokenizer.lang_code_to_id["en_XX"])
            doc=tokenizer.batch_decode(gtokens,skip_special_tokens=True)    
        context = {
            'title': 'Translated Output',
            'page_content': doc[0],
        }
        return render(request, 'pdf_document.html', context)
    else:
        return render(request, 'translate.html')

def ts(request):
    global doc
    languages = {'malayalam':'ml_IN', 'tamil':'ta_IN', 'hindi':'hi_IN', 'french':'fr_XX', 'japanese':'ja_XX' }
    if request.method == 'POST':
        lang = request.POST['lang']
        sumtype = request.POST['sumtype']
        model=MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
        tokenizer=MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50-many-to-many-mmt",src_lang=languages[lang])
        if request.POST['option']== 'upload':
            if request.FILES['pdf_document']:
                pdf_file = request.FILES['pdf_document']
                pdf_reader = PyPDF2.PdfReader(pdf_file)
                num_pages = len(pdf_reader.pages)
                page_content = []
                for page_num in range(num_pages):
                    page = pdf_reader.pages[page_num]
                    page_text = page.extract_text()
                    page_content.append(page_text)
                documents = ''
                for i in page_content:
                    documents += i
                model_inputs=tokenizer(documents,return_tensors='pt')
                gtokens=model.generate(**model_inputs,forced_bos_token_id=tokenizer.lang_code_to_id["en_XX"])
                doc=tokenizer.batch_decode(gtokens,skip_special_tokens=True)
        elif request.POST['option']== 'text':
            documents = request.POST['text']
            model_inputs=tokenizer(documents,return_tensors='pt')
            gtokens=model.generate(**model_inputs,forced_bos_token_id=tokenizer.lang_code_to_id["en_XX"])
            doc=tokenizer.batch_decode(gtokens,skip_special_tokens=True)    
        if sumtype == 'a':
            st = abssummarizer(doc[0])
            st = st[0]['summary_text']
        else:
            model = SBertSummarizer('paraphrase-MiniLM-L6-v2')
            st = model(doc[0], num_sentences=3)
        context = {
            'title': 'Translated and  Summarized Output',
            'page_content': st,
        }
        return render(request, 'pdf_document.html', context)
    else:
        return render(request, 'ts.html')

def hsum(request):
    if request.method == 'POST':
        documents = request.POST['input']
        t = request.POST['sumtype']
        if t == 'a':
            doc = abssummarizer(documents)
            doc = doc[0]['summary_text']
        elif t == 'e':
            model = SBertSummarizer('paraphrase-MiniLM-L6-v2')
            doc = model(documents, num_sentences=3)
        context = {
            'title': 'HuggingFace Summary Output',
            'page_content': doc,
        }
        return render(request, 'pdf_document.html', context)
    else:
        return render(request,'hugsum.html')

def output(request):
    return render(request, 'pdf_document.html')