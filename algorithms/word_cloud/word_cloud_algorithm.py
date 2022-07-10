from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
import pandas as pd
import nltk
from nltk.corpus import stopwords

df = pd.read_json('reviews_nubank.json')

comment_words = ''
stopwords = set(STOPWORDS)
stopwords.update(["a", "e", "i", "o", "u", "da", "em", "meu", "minha", "mais", "menos", "você", "de", "ao", "os", "para", "vai", "sim", "não", "que", "eu", "ele", "ela", "pra", "pro", "na", "no", "muito"])

# iterate through the json file
for val in df['text']:
	
	# typecaste each val to string
	val = str(val)

	# split the value
	tokens = val.split()
	
	# Converts each token into lowercase
	for i in range(len(tokens)):
		tokens[i] = tokens[i].lower()
	
	comment_words += " ".join(tokens)+" "

wordcloud = WordCloud(width = 800, height = 800,
				background_color ='white',
				stopwords = stopwords,
				min_font_size = 10).generate(comment_words)

# plot the WordCloud image					
plt.figure(figsize = (8, 8), facecolor = None)
plt.imshow(wordcloud)
plt.axis("off")
plt.tight_layout(pad = 0)

plt.show()
