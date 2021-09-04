import yfinance as yf
import streamlit as st
import nltk
import pandas as pd
import numpy as np
from st_aggrid import AgGrid, GridOptionsBuilder, DataReturnMode
pd.set_option("display.max_rows", None, "display.max_columns", None)

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('brown')
nltk.download('tagsets')
nltk.download('gutenberg')
nltk.download('genesis')
nltk.download('nps_chat')
nltk.download('inaugural')

# final_table = pd.DataFrame(columns=['Sentence','Score'])

def loadDataset():
	file = open('nltk.txt')
	read_file = file.read()
	text = nltk.Text(nltk.word_tokenize(read_file))
	tags = nltk.pos_tag(text)
	words = [word for (word, pos) in nltk.pos_tag(text)]
	pos = [pos for (word, pos) in nltk.pos_tag(text)]

	data = {'Words':words,'Pos':pos}
	df = pd.DataFrame(data)

	df['index']= df.index
	
	return df

def listSentencesPosition(data,separator):
    sentences = data.copy()
    sentences_borders = sentences[sentences['Words']==separator]
    
    return sentences_borders;

def pair_sentences_position_index(data):
    # Create pair of index (Begin,End) of each example/sentence
    splitedSize = 2
    splited_index = [data.index[x:x+splitedSize] for x in range(0, len(data.index), splitedSize)]
    
    return splited_index

def add_sentences_id(data,index_list):
    
    data["sentence_id"] = ""
    i = 1
    for coordinate in index_list:
    
        data['sentence_id'][(data['index']>=coordinate[0]) & (data['index']<=coordinate[1] )] = i
        i =i+1
    
    return data

#Find by ID the beginning and the end of example 
def get_example_position(data,sentencance_id):
    result = data[data['sentence_id']==sentencance_id]
    
    start = result.iloc[0]['index']
    end =  result.iloc[result.shape[0]-1]['index']
    
    return start,end
 
def add_score_column(data):
    # Adding score column and define it as numeric 
    data["Score"] = "1" #1 is the default value
    
    # Define "Score"and "Sentence_id" columns type as numerical 
    data = data.astype({"Score": int, "sentence_id": int})
    
    return data

def set_sentences_score(data):
    
    # Get unique id of examples
    unique_ids = data.sentence_id.unique()

    
    for sentence_id in unique_ids:
        
        #Get beginning and the end index of the example
        eg_start,eg_end = get_example_position(data,sentence_id)
        
        #Get the score which is always located at the end of the sentence
        score = data.iloc[eg_end-1]['Words']

        #attribute the score
        data.iloc[eg_start:eg_end+1,[4]] = int(score)
        
    return data

def convert_inputs_to_list(word1,word2):
    #Convert the input to a list to map with other type/part-of-speach of the word 
    word1 = [word1]
    word2 = [word2]
    
    return word1,word2

def lowercase_inputs(word1,word2):
    
    for i in range(len(word1)):
        word1[i] = word1[i].lower()

    for i in range(len(word2)):
        word2[i] = word2[i].lower()
        
    return word1,word2

def tag_input(input):

    if input =='verb':
        input = ['VBZ','VBP','VBG','VB','VBD','VBN']
    
    elif input == 'adjective' or input == 'adj'  :
        input = ['JJ','JJR','JJS','NN']
    
    elif input == 'adverb' or input == 'adv' :
        input = ['RB','RBR','RBS']
    
    elif input == 'noun' or input == 'pronoun' :
        input = ['NN','NNP','NNPS','PRP','NNS','PRP$']
    
    else:
        return[input]
    return input

def upperIndex(index,alist):
    #find the greater border index 
    position = np.argmax(alist > index)
    return alist[position]

def fliter_sentences(data,sentences_positions,word1,word2):
    
    #first fliter
    first_word_locations = data[(data['Words'].isin(word1)) | (data['Pos'].isin(word1))]
    first_word_locations['index'] = first_word_locations.index
    first_word_indices = first_word_locations["index"].values
    
    # if input 2 is not empty
    if not word2:
        #second filter
        good_first_input_indices = []
        for first_word_indice in first_word_indices:
            
            #Get the end of the example
            example_end_index = upperIndex(first_word_indice,sentences_positions.index)
            
            # Verifiy if the second input is close to the first one
            # at range of 4 words away
            if(first_word_indice+1 < example_end_index):
                tmp= data.apply(lambda x: data.iloc[first_word_indice+1].str.lower().isin(word2))
                if(tmp.Pos.Pos | tmp.Words.Words):
                    good_first_input_indices.append(first_word_indice)
            
            if(first_word_indice+2 < example_end_index):
                tmp= data.apply(lambda x: data.iloc[first_word_indice+2].str.lower().isin(word2))
                if(tmp.Pos.Pos | tmp.Words.Words):
                    good_first_input_indices.append(first_word_indice)
                    
            if(first_word_indice+3 < example_end_index):
                tmp= data.apply(lambda x: data.iloc[first_word_indice+3].str.lower().isin(word2))
                if(tmp.Pos.Pos | tmp.Words.Words):
                    good_first_input_indices.append(first_word_indice)
                    
            # if(first_word_indice+4 < example_end_index):
            #     tmp= data.apply(lambda x: data.iloc[first_word_indice+4].str.lower().isin(word2))
            #     if(tmp.Pos.Pos | tmp.Words.Words):
            #         good_first_input_indices.append(first_word_indice)
                    
    else:
        # if input 2 is empty we will take the result of input 1 
        good_first_input_indices = first_word_indices
    
    all_examples_indices = sentences_positions.index.array
    
    final_index_list = []
    for good_word1_index in good_first_input_indices:
        
        #get sentence id
        sentenance_id = data[data["index"]==good_word1_index]["sentence_id"].values[0]
        
        good_example_start,good_example_end = get_example_position(data,sentenance_id)
        
        final_index_list.append(good_example_start)
        final_index_list.append(good_example_end)
    
    #Couple the index to identifie sentence location
    splitedSize = 2
    final_index_list = [final_index_list[x:x+splitedSize] for x in range(0, len(final_index_list), splitedSize)]
    
    return final_index_list


def create_solution_dataframe(data,index_list):
    # Create the solution dataframe
    columns = ['Words','Pos']
    finalSolution = pd.DataFrame(columns=columns)
    
    for coordinate in index_list:
        finalSolution = finalSolution.append(data.iloc[coordinate[0]:coordinate[1]])
        
    return finalSolution


def dateframe_to_array(data,sep):
    
    # conver column to one string
    final_solution = ' '.join(data['Words'].values[1:])
    
    #Get examples scores
    scores = [int(s) for s in final_solution.split() if s.isdigit()]
    
    # Remove scores from the examples
    sentences = ''.join([i for i in final_solution if not i.isdigit()])

    # split the string
    sentences = sentences.split(";")
   
    result = {'Sentence':sentences,'Score': scores}
    
    result =pd.DataFrame(result)  

    return result

#----------------------------------------------------------------  
def main():
    # global final_table
    
    #Upload data set
    data = loadDataset()

    # localize sentenances beginning and end with the help of a sentence separator
    sentences_positions = listSentencesPosition(data,";")
    
    # pairing sentenances location index
    paired_index = pair_sentences_position_index(sentences_positions)
    
    #Adding sentences id
    data = add_sentences_id(data, paired_index)
    
    data = add_score_column(data)
    
    #Attribute sentences score
    data = set_sentences_score(data)
    
   
    form = st.form(key='my-form')
    first_word = form.text_input('First word')
    second_word = form.text_input('Second word')
    search_button = form.form_submit_button('Search')
    
    st.write('We will help you to improve your writing by using fancy words, please click on Search button')
    
    
    if search_button:
        #Convert inputs to lists for easier search
        first_word,second_word = convert_inputs_to_list(first_word, second_word)

        #Lowercase inputs
        first_word,second_word = lowercase_inputs(first_word,second_word)
        
        #Tag and map inputs
        first_word = tag_input(first_word[0])
        second_word = tag_input(second_word[0])
        
        #filter
        filterd_sentences_indices = fliter_sentences(data,sentences_positions, first_word, second_word)
        
        #Final solution dataframe
        final_solution = create_solution_dataframe(data,filterd_sentences_indices)
        
        #sort
        final_solution = final_solution.sort_values(by=['Score','sentence_id'],ascending=False)
        
        final_table = dateframe_to_array(final_solution,";")
        
        final_table = final_table.drop_duplicates()
        
        gb = GridOptionsBuilder.from_dataframe(final_table)
        gb.configure_grid_options(rowHeight=50)
        gb.configure_column('Score', editable=True, resizable=True,)
        
        grid_options = gb.build()
        grid_response = AgGrid(final_table, gridOptions=grid_options, data_return_mode='AS_INPUT', 
                                update_model='MODEL_CHANGE\D',width='100%',fit_columns_on_grid_load=True)
        
        # df = grid_response['data']
        # st.write(final_solution)
        # st.write(result.assign(hack='').set_index('hack'))
        
 
main()

# #Editable table 
# row = st.number_input('row', max_value=final_table.shape[0])
# col = st.number_input('column', max_value=final_table.shape[1])
# value = st.number_input('value')

# # Change the entry at (row, col) to the given value
# if(final_table.shape[0]>0 and final_table.shape[1]>0):
#     final_table.values[row][col] = value

# # And display the result!
# st.dataframe(final_table) 






