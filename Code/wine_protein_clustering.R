# clustering video

rm(list = ls())
library(ggplot2) # visualizations
library(reshape2)# for melting data frame (wide to long)

# ------ hierarchical example: protien consumption in Europe -------

food = read.csv('Data/protein.csv')
summary(food)
head(food)

# set up clustering (either type)
# create (understandardized) matrix of relevant, numeric values (everything but 1st)
food_X = food[, -1]
# standardize all predictor columns (we won't use column 1)
food_stand = apply(food_X, 2, function(x){(x - mean(x))/sd(x)})

# compute observation-observation distances (only hierarchical clustering)
food_dist = dist(food_stand, method = 'euclidean')

# first, let's use the average method to measure cluster-to-cluster
# similarity
food_clust = hclust(food_dist, method = 'average')
# the result from hierarchical clustering is a 'dendogram'
# the height of a bar is the distance between cluster cents
plot(food_clust, labels = food$Country, hang = -1)
# create clusters by drawing a horizontal line
# from notes: you cut the dendrogram tree with a horizontal line at a height where
# the line can traverse the maximum distance up and down without
# it intersectign the merging point
rect.hclust(food_clust, k = 3, border = 'red')

# let's also look at the ward method
food_clust = hclust(food_dist, method = 'ward.D')
plot(food_clust, labels = food$Country)
rect.hclust(food_clust, k = 5, border = 'red')

# making sense of the clusters: obtaining 'characterizations' of clusters
# save clusters
food_X$h_cluster = as.factor(cutree(food_clust, k = 5))
food_X_long = melt(food_X, id.vars = c('h_cluster'))
head(food_X_long)
ggplot(data = food_X_long) + 
  geom_boxplot(aes(x = h_cluster, y = value, fill = h_cluster)) + 
  facet_wrap(~ variable, scales = 'free') + 
  scale_fill_brewer("Cluster \nMembership", palette = 'Dark2') + 
  ggtitle('Hierarchical Clusters')

# name the clusters
# cluster 1: 'high-cereal diet'
# cluster 2: 'high meat-dairy diet'
# cluster 3: 'starchy diet'
# cluster 4: 'high fish and milk diet'
# cluster 5: 'high fish and plant diet'

# alternative way to visualize:
# save cluster labels as the row names of the original
rownames(food_stand) = food$Country
# look at heatmap
heatmap(as.matrix(food_stand), Colv = NA, col = paste('grey', 1:99, sep = ""))
# white = higher level of variable, black = lower level of variable

# ------ wine k-means example -----

rm(list = ls())
wine = read.csv('Data/wine.csv')
head(wine)
summary(wine)
# lots of characteristics measured here - use clustering to break into
# a small number of meaningful 'groups'

# setup for clustering (either type)
# collect relevant, numeric variables (everything)
wine_X = wine
# standardize all predictor columns (we won't use column 1)
wine_stand = apply(wine_X, 2, function(x){(x - mean(x))/sd(x)})

# k-means clustering --> must choose how many clusters we want
wss = (nrow(wine) - 1)*sum(apply(wine_stand, 2, var)) # wss for k = 1 (total ss of data)
for (i in 2:15) {wss[i] = sum(kmeans(wine_stand, centers = i)$withinss)}
# elbow method uses a scree plot:
plot(1:15, wss, type = 'b', xlab = 'Number of Clusters',
     ylab = 'Within Cluster of Sum of Squares')

# seems that k = 3 might be reasonable (this is subjective)
wine_kmeans = kmeans(wine_stand, 3)
str(wine_kmeans)

# show how distrubtions of each of the features vary across clusters
wine_X$km_cluster = as.factor(wine_kmeans$cluster)
wine_X_long = melt(wine_X, id.vars = 'km_cluster')
ggplot(data = wine_X_long) +
  geom_boxplot(aes(x = km_cluster, y = value, fill = km_cluster)) + 
  facet_wrap(~ variable, scales = 'free') + 
  scale_fill_brewer('Cluster \nMembership', palette = "Spectral") + 
  ggtitle('Hierarchical Clusters')











