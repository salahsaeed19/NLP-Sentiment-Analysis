# Restaurant Review Sentiment Analysis using Naive Bayes Classifier

### Introduction:
The objective of this project is to perform sentiment analysis on restaurant reviews using a Naive Bayes classifier. The goal is to develop a model that can accurately classify whether a review is positive or negative based on its text content.

### Dataset:
We used a dataset of restaurant reviews obtained from a TSV file. The dataset contains textual reviews along with their corresponding sentiment labels (positive or negative). It provides a valuable resource for training and evaluating our sentiment analysis model.

### Preprocessing:
Before building the model, we performed several preprocessing steps on the text data. These steps include:
* Removing non-alphabetic characters and special symbols from the reviews.
* Converting all text to lowercase to ensure consistency.
* Splitting the text into individual words.
* Removing stopwords, which are common words that do not carry much sentiment.
* Applying stemming using the Porter stemming algorithm to reduce words to their base form.

### Feature Extraction:
To convert the preprocessed text data into a numerical representation, we used the CountVectorizer from the scikit-learn library. The CountVectorizer tokenizes the text and counts the occurrences of each word, creating a matrix of token counts.

### Model Training:
We split the dataset into training and testing sets using a 67:33 ratio. The training set was used to train a Multinomial Naive Bayes classifier. The Naive Bayes algorithm is a popular choice for text classification tasks due to its simplicity and effectiveness.

### Model Evaluation:
After training the model, we evaluated its performance on the testing set. The accuracy score was calculated to measure the overall accuracy of the model in classifying the reviews. Additionally, we generated a classification report, which provides detailed metrics such as precision, recall, and F1-score for each sentiment class (positive and negative).

### Results:
The accuracy achieved by the Naive Bayes classifier on the testing set was 0.73, indicating that the model correctly classified 73% of the restaurant reviews. The classification report further revealed the performance of the model for each sentiment class, providing insights into precision, recall, and F1-score.

![‏‏لقطة الشاشة (352)](https://github.com/salahsaeed19/NLP-Sentiment-Analysis/assets/80893300/672d93db-63bb-4487-981b-fee8e732d483)



### Conclusion:
In this project, we successfully developed a sentiment analysis model using a Naive Bayes classifier to classify restaurant reviews as positive or negative. The model achieved a satisfactory accuracy score of 0.73 on the testing set, demonstrating its effectiveness in sentiment classification. This project highlights the importance of preprocessing text data, feature extraction, and the application of machine learning algorithms for sentiment analysis tasks.

### Future Enhancements:
To improve the model's performance, several enhancements can be considered. These include experimenting with different text preprocessing techniques, exploring alternative feature extraction methods such as TF-IDF, and experimenting with other classification algorithms. Additionally, incorporating more advanced techniques like word embeddings or deep learning models could potentially yield better results in sentiment analysis tasks.

Overall, this project serves as a foundation for sentiment analysis in the restaurant domain and can be further expanded and customized for more specific use cases or extended to other industries as well.
