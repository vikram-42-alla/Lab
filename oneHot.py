#Words
from tensorflow.keras.preprocessing.text import one_hot
words=["apple","cherry","orange","apple","cherry","orange"]
vocab=set(words)
word_to_int={word:i for i,word in enumerate(vocab)}
int_words=[word_to_int[word] for word in words]
one_hot_words=[]
for int_word in int_words:
  one_hot_word=[0]*len(vocab)
  one_hot_word[int_word]=1
  one_hot_words.append(one_hot_word)
print(one_hot_words)

#Characters
import string
input_string="hello world"
vocab=set(input_string)
char_to_int={char:i for i,char in enumerate(vocab)}
int_chars=[char_to_int[char] for char in input_string]
one_hot_chars=[]
for int_char in int_chars:
  one_hot_char=[0]*len(vocab)
  one_hot_char[int_char]=1
  one_hot_chars.append(one_hot_char)
print(one_hot_chars)
