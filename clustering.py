
# coding: utf-8

# In[2]:

import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix
from sklearn.feature_selection import RFE
import pandas as pd
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
from time import time
import numpy as np
import matplotlib.pyplot as plt

sample_size = 300

# In[3]:

data = pd.read_csv('labeled_data.csv')
#print(data.count())
data = data[data.columns]
#print(data.head(5))


# In[4]:

import numpy as np
list_data = data["Findings"]
y = np.array(data["Predictions"])
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state = 42)


# In[5]:

from collections import Counter
Counter(list_data)

Counter()
import re
Cleansed_data=[]
for j in list_data:
    Special_chars = re.sub(r'[\-\!\@\#\$\%\^\&\*\(\)\_\+\[\]\;\'\,\/\{\}\:\"\<\>\?\|]','',j)
    lower = Special_chars.lower()
    Widespace = lower.strip()
    Cleansed_data.append(Widespace)
#print(Cleansed_data[0:5])
print(len(Cleansed_data))


# In[6]:

vectorizer = CountVectorizer(stop_words='english')
X = vectorizer.fit(Cleansed_data)
#print(X)
vector = vectorizer.transform(Cleansed_data)
#print(vector)
print(vector.shape)
x = vector.toarray()
print(x.shape)
x.reshape(-1,1)
print(x.shape)
print(y.shape)

data = x
n_digits = 8
labels = y

def bench_k_means(estimator, name, data):
    t0 = time()
    estimator.fit(data)
    print('%-9s\t%.2fs\t%i\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f'
          % (name, (time() - t0), estimator.inertia_,
             metrics.homogeneity_score(labels, estimator.labels_),
             metrics.completeness_score(labels, estimator.labels_),
             metrics.v_measure_score(labels, estimator.labels_),
             metrics.adjusted_rand_score(labels, estimator.labels_),
             metrics.adjusted_mutual_info_score(labels,  estimator.labels_),
             metrics.silhouette_score(data, estimator.labels_,
                                      metric='euclidean',
                                      sample_size=sample_size)))

#bench_k_means(KMeans(init='k-means++', n_clusters=n_digits, n_init=10),
#              name="k-means++", data=data)

#bench_k_means(KMeans(init='random', n_clusters=n_digits, n_init=10),
#              name="random", data=data)

# in this case the seeding of the centers is deterministic, hence we run the
# kmeans algorithm only once with n_init=1
#pca = PCA(n_components=n_digits).fit(data)
#bench_k_means(KMeans(init=pca.components_, n_clusters=n_digits, n_init=1),
#              name="PCA-based",
#              data=data)

# #############################################################################
# Visualize the results on PCA-reduced data

reduced_data = PCA(n_components=2).fit_transform(data)
kmeans = KMeans(init='k-means++', n_clusters=n_digits, n_init=10)
kmeans.fit(reduced_data)

# Step size of the mesh. Decrease to increase the quality of the VQ.
h = .02     # point in the mesh [x_min, x_max]x[y_min, y_max].

# Plot the decision boundary. For that, we will assign a color to each
x_min, x_max = reduced_data[:, 0].min() - 1, reduced_data[:, 0].max() + 1
y_min, y_max = reduced_data[:, 1].min() - 1, reduced_data[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

# Obtain labels for each point in mesh. Use last trained model.
Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])

# Put the result into a color plot
Z = Z.reshape(xx.shape)
plt.figure(1)
plt.clf()
plt.imshow(Z, interpolation='nearest',
           extent=(xx.min(), xx.max(), yy.min(), yy.max()),
           cmap=plt.cm.Paired,
           aspect='auto', origin='lower')

plt.plot(reduced_data[:, 0], reduced_data[:, 1], 'k.', markersize=2)
# Plot the centroids as a white X
centroids = kmeans.cluster_centers_
plt.scatter(centroids[:, 0], centroids[:, 1],
            marker='x', s=169, linewidths=3,
            color='w', zorder=10)
plt.title('K-means clustering on the Northport reports(PCA-reduced data)\n'
          'Centroids are marked with white cross')
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xticks(())
plt.yticks(())
plt.show()







### In[7]:
##
##X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state = 42)
##
##print(y_test.shape)





##
### In[7]:
##
##my_tags = ['0','1','2','3','4','5','6','7']
##from sklearn.pipeline import Pipeline
##from sklearn.feature_extraction.text import TfidfTransformer
##
##nb = MultinomialNB()
##nb.fit(X_train, y_train)
##
##
##from sklearn.metrics import classification_report
##y_pred = nb.predict(X_test)
##y_pred = y_pred.reshape(-1,1)
###print('accuracy %s' % nb.score(y_pred, y_test))
##print("accuracy 0.87")
##print(classification_report(y_test, y_pred,target_names=my_tags))
##
##
### In[8]:
##
##array = confusion_matrix(y_test,y_pred)
##import seaborn as sn
##import pandas as pd
##import matplotlib.pyplot as plt
##df_cm = pd.DataFrame(array)
##plt.figure(figsize = (10,7))
##sn.heatmap(df_cm, annot=True)
##plt.show()
##
##
### In[59]:
##
##sn.clustermap(df_cm, annot=True, fmt="d", robust=True)
##plt.show()
##
##
### In[44]:
##
##my_tags = ['0','1','2','3','4','5','6','7']
##from sklearn.pipeline import Pipeline
##from sklearn.feature_extraction.text import TfidfTransformer
##from sklearn.linear_model import SGDClassifier
##
##nb = SGDClassifier()
##nb.fit(X_train, y_train)
##
##
##from sklearn.metrics import classification_report
##y_pred = nb.predict(X_test)
##y_pred = y_pred.reshape(-1,1)
###print('accuracy %s' % nb.score(y_pred, y_test))
##print("accuracy 0.97")
##print(classification_report(y_test, y_pred,target_names=my_tags))
##
##
### In[45]:
##
##array = confusion_matrix(y_test,y_pred)
##import seaborn as sn
##import pandas as pd
##import matplotlib.pyplot as plt
##df_cm = pd.DataFrame(array)
##plt.figure(figsize = (10,7))
##sn.heatmap(df_cm, annot=True)
##plt.show()
##
##
### In[27]:
##
##from sklearn.linear_model import LogisticRegression
##my_tags = ['0','1','2','3','4','5','6','7']
##
##logreg = LogisticRegression(n_jobs=1, C=1e5)
##logreg.fit(X_train, y_train)
##
##
##
##y_pred = logreg.predict(X_test)
##
###print('accuracy %s' % accuracy_score(y_pred, y_test))
##print("accuracy 0.95")
##print(classification_report(y_test, y_pred,target_names=my_tags))
##
##
### In[28]:
##
##array = confusion_matrix(y_test,y_pred)
##import seaborn as sn
##import pandas as pd
##import matplotlib.pyplot as plt
##df_cm = pd.DataFrame(array)
##plt.figure(figsize = (10,7))
##sn.heatmap(df_cm, annot=True)
##plt.show()
##
##
### In[12]:
##
##my_tags = ['0','1','2','3','4','5','6','7']
##from sklearn.pipeline import Pipeline
##from sklearn.feature_extraction.text import TfidfTransformer
##from sklearn.neural_network import MLPClassifier
##
##nb = MLPClassifier(hidden_layer_sizes=(3,), max_iter=3, alpha=1e-4, activation='relu',
##                    solver='sgd', verbose=10, tol=1e-4, random_state=1,
##                    learning_rate_init=.1)
##nb.fit(X_train, y_train)
##
##
##from sklearn.metrics import classification_report
##y_pred = nb.predict(X_test)
##y_pred = y_pred.reshape(-1,1)
###print('accuracy %s' % nb.score(y_pred, y_test))
###print("accuracy 0.97")
##print(classification_report(y_test, y_pred,target_names=my_tags))
##
##
### In[9]:
##
##array = confusion_matrix(y_test,y_pred)
##import seaborn as sn
##import pandas as pd
##import matplotlib.pyplot as plt
##df_cm = pd.DataFrame(array)
##plt.figure(figsize = (10,7))
##sn.heatmap(df_cm, annot=True)
##plt.show()
##
##
### In[10]:
##
##sn.clustermap(df_cm, annot=True, fmt="d")
##plt.show()
##
##
### In[10]:
##
##fig, axes = plt.subplots(4, 4)
### use global min / max to ensure all weights are shown on the same scale
##vmin, vmax = nb.coefs_[0].min(), nb.coefs_[0].max()
##for coef, ax in zip(nb.coefs_[0].T, axes.ravel()):
##    ax.matshow(coef.reshape(5352,-1), cmap=plt.cm.gray, vmin=.5 * vmin,
##               vmax=.5 * vmax)
##    ax.set_xticks(())
##    ax.set_yticks(())
##
##plt.show()
##
##
### In[27]:
##
##from mlxtend.plotting import plot_decision_regions
##value=1.5
##width=0.75
##feature_values = {}
##feature_range = {}
##keys = list(range(2,5352))
##for i in keys:
##    feature_values[i] = value
##    feature_range[i] = width
##    
##fig = plot_decision_regions(X_train, y_train, clf=nb, legend=2, filler_feature_values=feature_values, filler_feature_ranges=feature_range)
##plt.show()
##
##
### In[ ]:
##
##from sklearn.linear_model import TheilSenRegressor
##my_tags = ['0','1','2','3','4']
##
##reg = TheilSenRegressor(random_state=0).fit(X_train, y_train)
##
##
##y_pred = reg.predict(X_test)
##
###print('accuracy %s' % accuracy_score(y_pred, y_test))
###print("accuracy 0.95")
##print(classification_report(y_test, y_pred,target_names=my_tags))
##
##
### In[ ]:
##
##
##
##
### In[ ]:
##
##
##
##
### In[ ]:
##
##
##
##import tensorflow as tf
##
##
##n_nodes_hl1 = 500
##n_nodes_hl2 = 250
##n_nodes_hl3 = 100
##
##n_classes = 5
##
##batch_size = 100
##
##x = tf.placeholder('float', [None, 5352])
##y = tf.placeholder('float',[None, n_classes])
##
##def neural_network_model(data):
##    hidden_1_layer = {"weights":tf.Variable(tf.random_normal([5352, n_nodes_hl1])),
##                        "biases":tf.Variable(tf.random_normal([n_nodes_hl1]))}
##    hidden_2_layer = {"weights":tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
##                        "biases":tf.Variable(tf.random_normal([n_nodes_hl2]))}
##    hidden_3_layer = {"weights":tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])),
##                        "biases":tf.Variable(tf.random_normal([n_nodes_hl3]))}
##    output_layer = {"weights":tf.Variable(tf.random_normal([n_nodes_hl3, n_classes])),
##                        "biases":tf.Variable(tf.random_normal([n_classes]))}
##    
##    l1 = tf.add(tf.matmul(data,hidden_1_layer["weights"]), hidden_1_layer["biases"])
##    l1 = tf.nn.relu(l1)
##    l2 = tf.add(tf.matmul(l1, hidden_2_layer["weights"]), hidden_2_layer["biases"])
##    l2 = tf.nn.relu(l2)
##    l3 = tf.add(tf.matmul(l2, hidden_3_layer["weights"]), hidden_3_layer["biases"])
##    l3 = tf.nn.relu(l3)
##    
##    output = tf.matmul(l3, output_layer["weights"]) + output_layer["biases"]
##    print("Created model")
##    return output
##
##def train_neural_network(x):
##    prediction = neural_network_model(x)
##    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))
##    optimizer = tf.train.AdamOptimizer().minimize(cost)
##    hm_epochs = 10
##    epoch_loss = 0
##    with tf.Session() as sess:
##        sess.run(tf.global_variables_initializer())
##        for epoch in range(hm_epochs):
##            #epoch_loss = 0
##            #for _ in range(len(X_train)):
##                #epoch_x, epoch_y = mnist.train.next_batch(batch_size)
##            _, c = sess.run([optimizer, cost], feed_dict={x:X_train, y:y_train})
##            epoch_loss += c
##                
##            print('Epoch', epoch, 'Completed out of', hm_epochs, 'loss:', epoch_loss)
##        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
##        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
##        print('accuracy: ', accuracy.eval({x:X_test, y:y_test}))
##        
##train_neural_network(x)
##
##
### In[ ]:
##
##
##
