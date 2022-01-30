library(tidyverse)
library(gmodels)
library(tm)
library(wordcloud)
library(e1071)
library(caret)
library(keras)
library(tensorflow)

dfIMDB <- read.csv("movie_reviews.csv")

colnames(dfIMDB) <- c("text", "label")
View(dfIMDB)

set.seed(1985)
dfIMDB <- dfIMDB[order(runif(n=30000)),]

### 50/50 when it comes to labels
round(prop.table(table(dfIMDB$label)),2)

### Getting rid of all the HTML tags
dfIMDB$text <- gsub("<[^>]+>", "", dfIMDB$text)

corpus = VCorpus(VectorSource(dfIMDB$text))
### Checking the first movie review before Data Cleaning
as.character(corpus[[1]])

corpus = tm_map(corpus, content_transformer(tolower))
corpus = tm_map(corpus, removeNumbers)
corpus = tm_map(corpus, removePunctuation)
corpus = tm_map(corpus, removeWords, stopwords("english"))
corpus = tm_map(corpus, stemDocument)
corpus = tm_map(corpus, stripWhitespace)
as.character(corpus[[1]])

dtm = DocumentTermMatrix(corpus)
dtm
dim(dtm)
dtm = removeSparseTerms(dtm, 0.999)
dtm
dim(dtm)

#Generating the word clouds
negative <- subset(dfIMDB, label == 0)
positive <- subset(dfIMDB, label == 1)

head(negative)
head(positive)

wordcloud(negative$text, max.words = 100, scale = c(3,0.5))
wordcloud(positive$text, max.words = 100, scale = c(3,0.5))


### Most used terms in movie reviews
freq<- sort(colSums(as.matrix(dtm)), decreasing=TRUE)
library(ggplot2)
wf<- data.frame(word=names(freq), freq=freq)
head(wf)

wordcloud(names(freq), max.words = 100, scale = c(3,0.5))

### Converting to categorical values
convertCategorical <- function(x) {
  y <- ifelse(x > 0, 1,0)
  y <- factor(y, levels=c(0,1), labels=c("No", "Yes"))
  y
}

dfCategorical <- apply(dtm, 2, convertCategorical)
dataset = as.data.frame(as.matrix(dfCategorical))

dataset$Class = as.factor(dfIMDB$label)
str(dataset$Class)

head(dataset)
dim(dataset)

set.seed(222)
split = sample(2,nrow(dataset),prob = c(0.75,0.25),replace = TRUE)
train_set = dataset[split == 1,]
test_set = dataset[split == 2,] 

prop.table(table(train_set$Class))
prop.table(table(test_set$Class))

### Naive Bayes Algorithm
text.classifer <- naiveBayes(train_set, train_set$Class)
text.pred <- predict(text.classifer, test_set)

CrossTable(text.pred, test_set$Class,
           prop.chisq = FALSE, 
           prop.t = FALSE,
           dnn = c('predicted', 'actual'))

confusionMatrix(table(text.pred, test_set$Class))

### Random Forest model
library(randomForest)
rf_classifier = randomForest(x = train_set,
                             y = train_set$Class,
                             ntree = 300)

rf_pred = predict(rf_classifier, newdata = test_set)

confusionMatrix(table(rf_pred, test_set$Class))

### Log Regression
#model <- glm( Class ~., data = train_set, family = binomial)
#probabilities <- model %>% predict(test.data, type = "response")
#predicted.classes <- ifelse(probabilities > 0.5, 1, 0) #1 for pos, 0 for neg

### NN

#model <- keras_model_sequential()
#model %>%
#  layer_embedding(input_dim = max_features,
#                  output_dim = 128,
#                  input_length = maxlen) %>%
#  bidirectional(layer_lstm(units = 64)) %>%
#  layer_dropout(rate = 0.5) %>%
#  layer_dense(units = 1, activation = 'sigmoid')

#model %>% compile(
#  loss = 'binary_crossentropy',
#  optimizer = 'adam',
#  metrics = c('accuracy')
#)

#model %>% fit(
#  train_set, train_set$Class,
#  batch_size = batch_size,
#  epochs = 4,
#  validation_data = list(test_set, test_set$Class)
#)


