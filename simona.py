#Simona Halep Interview Analysis

#Imported the necessary packeges for the first stpes
import requests
from bs4 import BeautifulSoup

# We use the requests package to get the HTML code
URL = 'https://www.euronews.com/2023/12/15/i-know-im-clean-two-time-grand-slam-winner-simona-halep-opens-up-about-doping-scandal'
page = requests.get(URL)

# We use the BeautifulSoup package to extract the HTML
soup = BeautifulSoup(page.content, 'html.parser')

# We find all <strong> tags with the text 'Simona Halep'
halep_strong_tags = soup.find_all('strong', string='Simona Halep')

# We make an empty list to hold Simona's answers
halep_answers = []
for strong_tag in halep_strong_tags:
    # Looking for the answers
    next_sibling = strong_tag.next_sibling
    while next_sibling and next_sibling.name not in ['strong', 'h2', 'h3']:  # Assuming new questions are in these tags
        if next_sibling.string:
            halep_answers.append(next_sibling.string.strip())
        next_sibling = next_sibling.next_sibling

# Cleaning the extracted the answers
cleaned_answers = [' '.join(answer.split()) for answer in halep_answers if answer and not answer.isspace()]


# Cleaning the text completely
halep_speech_str = ""
for answer in halep_answers:
    cleaned_answer = answer.replace('[:', '').replace(':",', '').replace(':', '').replace(']', '').strip()
    halep_speech_str += " " + cleaned_answer

import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')


from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# Define English stop words, tokanize the answers and convert, filter stop words
stop_words = stopwords.words('english')
words = word_tokenize(halep_speech_str.lower())
filtered_speech = [w for w in words if w.isalpha() and not w in stop_words]

#Analyze the frequency
from nltk import FreqDist
freq = FreqDist(filtered_speech)
freq.most_common(20)

from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

lemmatized_text = [lemmatizer.lemmatize(word) for word in filtered_speech]

freq_lemma = FreqDist(lemmatized_text)
freq_lemma.most_common(20)   
    
# Print the 10 most common words
for word, frequency in freq_lemma.most_common(10):
    print(f"{word}: {frequency}")

#Creating the bar graph
import plotly.express as px
common_words, common_freqs = zip(*freq_lemma.most_common(10))

fig = px.bar(x = common_freqs, y = common_words, orientation = 'h')
fig.update_layout(yaxis = dict(autorange="reversed"))

#Download bar graph to our folder
file_path = '/Users/andi/Desktop/proiect/bargraph.png'
fig.write_image(file_path)
# print(f"Figure saved as {file_path}")


import re

# Make the Regex cleaning
expression = "[^a-zA-Z0-9 ]"  # keep only letters, numbers, and whitespace
cleantextCAP = re.sub(expression, ' ', halep_speech_str)  #apply
cleantext = cleantextCAP.lower() # lower case 


# Save for wordcloud
with open("Output.txt", "w") as text_file:
    text_file.write(cleantext)

# Count and create dictionary
words = cleantext.split(" ")
print((len(words)))

dict1 = {}
for word in words:
    if word:  
        dict1[word] = words.count(word)

print(cleantext)

from  nltk.corpus import stopwords

# Unsorted speech constituents in dictionary as dict1
keys = list(dict1)
filtered_words = [word for word in keys if word not in stopwords.words('english')]
dict2 = dict((k, dict1[k]) for k in filtered_words)

filtered_words

# Resort in list
# Reconvert to dictionary
def SequenceSelection(dictionary, length, startindex = 0): # length is length of highest consecutive value vector
    
    # Test input
    lengthDict = len(dictionary)
    if length > lengthDict:
        return print("length is longer than dictionary length");
    else:
        d = dictionary
        items = [(v, k) for k, v in d.items()]
        items.sort()
        items.reverse()
        itemsOut = [(k, v) for v, k in items]

        highest = itemsOut[startindex: startindex + length]
        dd = dict(highest)
        wanted_keys = dd.keys()
        dictshow = dict((k, d[k]) for k in wanted_keys if k in d)

        return dictshow;

dictshow = SequenceSelection(dictionary = dict2, length = 7, startindex = 0)

dictshow

range(len(dictshow))

import matplotlib.pyplot as plt

# Plot most frequent words
n = range(len(dictshow))
plt.figure()
plt.bar(n, dictshow.values(), align='center')
plt.xticks(n, dictshow.keys())
plt.title("Most frequent Words")
plt.savefig("FrequentWords.png", transparent=True)

from os import path
import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS

root_path = os.getcwd()

# Read the whole text.
with open(path.join(root_path, 'Output.txt'), 'r', encoding='utf-8', errors='ignore') as output_file:
    text = output_file.readlines()


# Optional additional stopwords
stopwords = set(STOPWORDS)
stopwords.add("said")
stopwords.add("s")
stopwords.add("m")
stopwords.add("ve")

# Construct Word Cloud
# no backgroundcolor and mode = 'RGBA' create transparency
wc = WordCloud(max_words=1000, stopwords=stopwords, mode='RGBA', background_color=None)

# Pass Text
wc.generate(text[0])

# store to file
wc.to_file(path.join(root_path, "simonahalep.png"))

# show
plt.figure()
plt.imshow(wc, interpolation='bilinear')
plt.axis("off")
plt.imshow(wc, cmap=plt.cm.gray, interpolation='bilinear')
plt.axis("off")
plt.show()


#Cool Mask
from os import path
import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS

root_path = os.getcwd()

# Read the whole text.
with open(path.join(root_path, 'Output.txt'), 'r', encoding='utf-8', errors='ignore') as output_file:
    text = output_file.readlines()

# Mask
mask = np.array(Image.open(path.join(root_path, "simonahalep_mask.jpg")))

# Optional additional stopwords
stopwords = set(STOPWORDS)
stopwords.add("said")
stopwords.add("s")
stopwords.add("m")
stopwords.add("ve")

# Construct Word Cloud
# no backgroundcolor and mode = 'RGBA' create transparency
wc = WordCloud(max_words=1000, stopwords=stopwords, mode='RGBA', background_color=None, mask=mask)

# Pass Text
wc.generate(text[0])

# show
plt.figure()
plt.imshow(wc, interpolation='bilinear')
plt.axis("off")
plt.imshow(wc,cmap=plt.cm.gray, interpolation='bilinear')
plt.axis("off")
plt.savefig("simonahalep_maskcloud.png")
plt.show()


from textblob import TextBlob

sentiment = TextBlob(cleantext)
print("Polarity Score: ", sentiment.sentiment.polarity)

print("Subjectivity Score: ", sentiment.sentiment.subjectivity)



























    
    
    
    
    
    
    
    
    
