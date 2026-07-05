import nltk

paragraph= """There are broadly two types of extractive summarization tasks depending on what the summarization program focuses on. The first is generic summarization, which focuses on obtaining a generic summary or abstract of the collection (whether documents, or sets of images, or videos, news stories etc.). The second is query relevant summarization, sometimes called query-based summarization, which summarizes objects specific to a query. Summarization systems are able to create both query relevant text summaries and generic machine-generated summaries depending on what the user needs.
An example of a summarization problem is document summarization, which attempts to automatically produce an abstract from a given document. Sometimes one might be interested in generating a summary from a single source document, while others can use multiple source documents (for example, a cluster of articles on the same topic). This problem is called multi-document summarization. A related application is summarizing news articles. Imagine a system, which automatically pulls together news articles on a given topic (from the web), and concisely represents the latest news as a summary.
Image collection summarization is another application example of automatic summarization. It consists in selecting a representative set of images from a larger set of images.[4] A summary in this context is useful to show the most representative images of results in an image collection exploration system. Video summarization is a related domain, where the system automatically creates a trailer of a long video. This also has applications in consumer or personal videos, where one might want to skip the boring or repetitive actions. Similarly, in surveillance videos, one would want to extract important and suspicious activity, while ignoring all the boring and redundant frames captured """


#cleaning the texts
import re #re library will use for regular expression
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer

ps=PorterStemmer()
wordnet=WordNetLemmatizer()
sentences=nltk.sent_tokenize(paragraph)

corpus=[]

#create a empty list as corpus beacuse after cleaned the data corpus will store this clean data


for i in range(len(sentences)):
    review=re.sub('[^a-zA-Z]',' ',sentences[i])
    review=review.lower()
    review=review.split()
#    review=[ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review=[wordnet.lemmatize(word) for word in review if not word in set(stopwords.words('english'))]
    review=' '.join(review)
    corpus.append(review)
    
#creating teh Bag of words model

#Also we called as document matrix




from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()
X_bow=cv.fit_transform(corpus).toarray()

from sklearn.feature_extraction.text import TfidfVectorizer
tf=TfidfVectorizer()
X_tf=tf.fit_transform(corpus).toarray()