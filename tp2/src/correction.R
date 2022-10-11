
# Correction de la partie 2 du TD 2

# Set-up ------------------------------------------------------------------------------------

# install.packages("CHAID", repos="http://R-Forge.R-project.org")
library('CHAID') # modèle CHAID
library('rpart') # modèle CART
library('rpart.plot') # visualisation du modèle CART


# Question 9 - Un premier modèle ---------------------------------------------------------

# Importer les données
musch <- read.csv(file = "tp2/data/muschroom.csv", stringsAsFactors = TRUE)

# Enlever la colonne X
musch <- musch[, !names(musch) == "X"]

# Définir les échantillons train et test
train <- musch[musch$echantillon == "base", !names(musch) == "echantillon"]
test <- musch[musch$echantillon == "test", !names(musch) == "echantillon"]

# Définir les paramètres utilisés par rpart
cart_parameters <- rpart.control(
  minsplit = 60, # il faut au moins 60 observations dans un noeud pour le diviser
  minbucket = 30, # une division ne doit pas générer un noeud avec moins de 30 observations
  xval = 10, # nombre de blocs utilisés pour la validation croisée de l'élagage
  maxcompete = 4, # nombre de divisions compétitives retenues (equi reducteur)
  maxsurrogate = 4, # nombre de divisions surrogates retenues (equi divisant)
  usesurrogate = 2, # comment sont gérées les valeurs manquantes, voir la documentation pour plus d'infos
  maxdepth = 30, # la profondeur maximal de l'arbre,
  cp = 0 # Ne pas limiter la construction de l'arbre maximal
)

# Entraînement du modèle
cart_model <- rpart(
  formula = classe ~ .,
  data = train,
  method = "class",
  control = cart_parameters,
  parms = list(split = 'gini')
)

# Effectuer une prédiction sur les données test
cart_pred <- predict(
  object = cart_model,
  newdata = test,
  type = "class"
)

# Visualiser l'arbre
rpart.plot(cart_model)

# Afficher la matrice de confusion
table(test$classe, cart_pred)

# Sélection de l'arbre avec l'erreur la plus faible
split_best_xerror <- which.min(cart_model$cptable[, 4])

# Ajout de l'écart-type pour obtenir la borne supérieure de l'erreur à ne pas dépasser
min_error_std <- cart_model$cptable[split_best_xerror, 4] + cart_model$cptable[split_best_xerror, 5]

# Sélection de tous les arbres dont l'erreur est inféreiure au seuil calculé plus haut
possible_trees <- cart_model$cptable[cart_model$cptable[, 4] <= min_error_std, , drop = F]

# Ordonner les arbres selon leur complexité (nombre de splits)
selected_tree <- possible_trees[order(possible_trees[, 2]),, drop = F][1, ]

# Choix de l'arbre le plus simple
cp_optim <- selected_tree[1]

# Elaguer l'arbre
cart_model_pruned <- prune(cart_model, cp = cp_optim)

# Visualiser l'arbre
rpart.plot(cart_model_pruned)

# Effectuer une prédiction sur les données test
cart_pred <- predict(
  object = cart_model_pruned,
  newdata = test,
  type = "class"
)

# Afficher la matrice de confusion
table(test$classe, cart_pred)

# Taux d'erreur
error_rate <- sum(cart_pred != test$classe) / nrow(test)

summary(cart_model)
printcp(cart_model)


# Adding costs --------------------------------------------------------------------------------

# Adding special cost for errors: C(eatable/poison) = 1000, C(poison/eatable) = 1

# build the model
cart_model_cost <- rpart(
  classe ~ .,
  data = train,
  parms = list(
    split = "gini",
    loss = matrix(
      c(0, 1, 1000, 0),
      byrow = TRUE,
      nrow = 2)
  ),
  control = cart_parameters
)

# print the model
print(cart_model_cost)

# plot the tree
rpart.plot(cart_model_cost)

# print the results of the model
summary(cart_model_cost)

# evolution of the tree size and of the error based on the cp parameter
plotcp(model2)

# probabilities for the two classes for each observations of the test set
predict(model2, test)

# predicted class for each observtion
predtest2 <- predict(model2, test, type = "class")

# confusion matrix
table(test$classe, predtest2)

# calcul du taux d'erreur en test
sum(predtest2 != test$classe) / nrow(test)


# CHAID ---------------------------------------------------------------------------------------

# Decision tree using CHAID

# see the help page
?chaid

# chaid decision tree
model3 <- ctree(
  formula = classe ~ .,
  data = train
)

# model results
print(model3)

# plot the model
plot(
  model3,
  uniform = TRUE,
  compress = TRUE,
  margin = 0.2,
  branch = 0.3
)


