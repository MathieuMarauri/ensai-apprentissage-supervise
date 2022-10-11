
# Discimination - TP3 : comparaison de méthodes.

# L'objectif est de comparer différents modèles sur la prédiction de spam à l'aide du jeu de données
# spambase.

# Set up --------------------------------------------------------------------------------------

# Chargement des librairies nécessaires pour le TP.

# install.packages("caret")
# install.packages("class")
# install.packages("klaR")
# install.packages("dplyr")
# install.packages("e1071")
# install.packages("ggplot2")
# install.packages("MASS")
# install.packages("ROCR")
# install.packages("rpart")
# install.packages("scales")

library("caret") # ensemble de fonctions nécessaire au processus d'apprentissage supervisé
library("class") # fonction knn
library("klaR") # classifieur bayésien naïf (charger avant dplyr pour éviter problème avec select)
library("dplyr") # dataset manipulation
library("e1071") # fonction tune.knn
library("ggplot2") # visualisation
library("MASS") # analyse disciminante linéaire
library("ROCR") # courbe ROC
library("rpart") # cart
library("scales") # modifications des axes des graphes

# Set default ggplot theme
theme_set(theme_light(base_size = 20))


# Préparation du modèle -----------------------------------------------------------------------

# Chargement de la table spambase, construction des ensembles de train et de test.

# importer la table spambase dans l'environnement
spambase <- read.csv('tp3/data/spambase.csv', stringsAsFactors = FALSE)

# transformation de la variable cible en factor
spambase <- spambase %>%
  mutate(is_spam = as.factor(is_spam))

# construction des train et test sets, le test set sera utilisé pour comparer les modèles
train_set <- sample(nrow(spambase), (4 * nrow(spambase)) / 5)
test_set <- setdiff(1:nrow(spambase), train_set)
spambase_train <- spambase[train_set, ]
spambase_test <- spambase[test_set, ]


# KNN -----------------------------------------------------------------------------------------

# Construction du modèle de Knn avec sélection du meilleur K.
#
# relancer la cross validation plusieurs fois pour déterminer le meilleur K (la valeur peut varier
# pour chaque lancement de tune.knn)
best_k <- numeric(length = 10)
for (i in 1:10) {
  knn_cross <- tune.knn(
    x = spambase_train %>% select(-is_spam),
    y = spambase_train$is_spam,
    k = 1:10,
    tunecontrol = tune.control(sampling = "cross"),
    cross = 5
  )
  best_k[i] <- knn_cross$best.parameters
  print(i)
}
summary(unlist(best_k))

# on va ensuite tester la qualité du modèle avec le paramètre choisi (k = 1)
set.seed(123456)
knn_pred <- knn(
  train = spambase_train %>% select(-is_spam) %>% as.matrix(), # données d'apprentissage
  test = spambase_test %>% select(-is_spam) %>% as.matrix(), # données à prédire
  cl = spambase_train$is_spam, # vraies valeurs
  k = 1 # nombre de voisins
)

# matrice de confusion, taux de bonnes précisions et taux d'erreur
table(knn_pred, spambase_test$is_spam)
sum(knn_pred == spambase_test$is_spam) / nrow(spambase_test)
sum(knn_pred != spambase_test$is_spam) / nrow(spambase_test)


# Naive Bayes ---------------------------------------------------------------------------------

# Construction du modèle de Naive Bayes avec sélection des meilleurs hyper paramètres.

# définir la liste des hyperparamètres possibles pour le classifieur bayésien naïf
grid <- expand.grid(
  usekernel = c(TRUE, FALSE), # si vrai utilisation d'un noyau sinon gaussien
  fL = 0, # correction avec lissage de Laplace
  adjust = 1:5 # bandwidth
)
grid <- grid[!(grid$usekernel == FALSE & grid$adjust %in% 2:5), ]

# définir la méthode de validation, ici 10-fold cross validation
control <- trainControl(method = "cv", number = 5)

# on optimise les paramètres du modèle
nb <- train(
  x = spambase_train %>% select(-is_spam), # prédicteurs
  y = spambase_train$is_spam, # réponse
  method = "nb", # classifieur utilisé, ici Naive Bayes
  trControl = control, # méthode d'échantillonage, ici 10-fold CV
  tuneGrid = grid # liste des paramètres à comparer
)

# visualisation des résultats
plot(nb)

# prédiction avec le meilleur modèle
nb_pred <- predict(nb, spambase_test)

# matrice de confusion, taux de bonnes précisions et taux d'erreur
table(nb_pred, spambase_test$is_spam)
sum(nb_pred == spambase_test$is_spam) / nrow(spambase_test)
1 - sum(nb_pred == spambase_test$is_spam) / nrow(spambase_test)


# CART ----------------------------------------------------------------------------------------

# Construction du modèle CART.

# construction de l'arbre
cart <- rpart(
  formula = is_spam ~ .,
  data = spambase_train,
  method = "class",
  parms = list(split = "gini")
)

# prédiction des nouvelles valeurs sur le test set
cart_pred <- predict(cart, newdata = spambase_test, type = "class")

# matrice de confusion, taux de bonnes précisions et taux d'erreur
table(cart_pred, spambase_test$is_spam)
sum(cart_pred == spambase_test$is_spam) / nrow(spambase_test)
sum(cart_pred != spambase_test$is_spam) / nrow(spambase_test)


# LDA ----------------------------------------------------------------------------------------

# Construction du modèle d'analyse discriminante linéaire.

# construction du modèle
lda <- lda(is_spam ~ ., data = spambase_train)

# prédition sur les données de test
lda_pred <- predict(lda, spambase_test)

# matrice de confusion, taux de bonnes précisions et taux d'erreur
table(lda_pred$class, spambase_test$is_spam)
sum(lda_pred$class == spambase_test$is_spam) / nrow(spambase_test)
sum(lda_pred$class != spambase_test$is_spam) / nrow(spambase_test)


# Comparaison des méthodes --------------------------------------------------------------------

# Comparaison des méthodes KNN, CART, Naive Bayes et LDA. Utilisation de la courbe ROC, de l'AUC, du
# taux d'erreur global, des taux de vrais et faux positifs, ...

# Récupération des probas pour la classe 1 (spam) pour chaque modèle (sauf knn qui n'a pas de sens)
?prediction

prob_lda <- predict(lda, spambase_test)$posterior[, 2] %>% prediction(spambase_test$is_spam)
prob_cart <- predict(cart, spambase_test)[, 2] %>% prediction(spambase_test$is_spam)
prob_nb <- predict(nb, spambase_test, type = "prob")[, 2] %>% prediction(spambase_test$is_spam)

# Génération des points pour construction de la courbe ROC
?performance

roc_lda <- performance(prob_lda, "tpr", "fpr")
roc_cart <- performance(prob_cart, "tpr", "fpr")
roc_nb <- performance(prob_nb, "tpr", "fpr")

# Construction des data.frames nécessaires à l'affichage des coubres
roc_lda <- data.frame(
  "fpr" = roc_lda@x.values[[1]],
  "tpr" = roc_lda@y.values[[1]],
  "seuil" = roc_lda@alpha.values[[1]]
)
roc_cart <- data.frame(
  "fpr" = roc_cart@x.values[[1]],
  "tpr" = roc_cart@y.values[[1]],
  "seuil" = roc_cart@alpha.values[[1]]
)
roc_nb <- data.frame(
  "fpr" = roc_nb@x.values[[1]],
  "tpr" = roc_nb@y.values[[1]],
  "seuil" = roc_nb@alpha.values[[1]]
)

# Courbes ROC des 3 modèles
ggplot(
  data = roc_lda,
  mapping = aes(x = fpr, y = tpr)
) +
  # courbe du modèle LDA
  geom_line(
    mapping = aes(color = "LDA")
  ) +
  # courbe du modèle cart
  geom_line(
    data = roc_cart,
    mapping = aes(x = fpr, y = tpr, color = "CART")
  ) +
  # courbe du modèle NB
  geom_line(
    data = roc_nb,
    mapping = aes(x = fpr, y = tpr, color = "Naive Bayes")
  ) +
  # ajout du modèle aléatoire
  geom_segment(
    mapping = aes(x = 0, y = 0, xend = 1, yend = 1),
    linetype = "dashed",
    size = .2
  ) +
  # ajout du meilleur modèle
  geom_segment(
    mapping = aes(x = 0, y = 0, xend = 0, yend = 1),
    linetype = "dashed",
    size = .2
  ) +
  geom_segment(
    mapping = aes(x = 0, y = 1, xend = 1, yend = 1),
    linetype = "dashed",
    size = .2
  ) +
  # valeurs des axes en pourcentage
  scale_x_continuous(labels = percent) +
  scale_y_continuous(labels = percent) +
  # définition des couleurs des courbes
  scale_colour_manual(
    values = c("CART" = "#1D88E5", "Naive Bayes" = "#FFC108", "LDA" = "#D81A60")
  ) +
  # définition des noms des axes, légende et titre
  labs(
    x = "Taux de faux positifs",
    y = "Taux de vrais positifs",
    title = "Courbes ROC",
    colour = "Modèle"
  ) +
  # position de la légende sur le graphe et non pas à côté
  theme(
    legend.position = c(.8, .3)
  )

# AUC
performance(prob_lda, "auc")@y.values[[1]]
performance(prob_cart, "auc")@y.values[[1]]
performance(prob_nb, "auc")@y.values[[1]]

# Génération des points pour construction de la courbe LIFT
lift_lda <- performance(prob_lda, "sens","rpp")
lift_cart <- performance(prob_cart, "sens","rpp")
lift_nb <- performance(prob_nb, "sens","rpp")

# Construction des data.frames nécessaires à l'affichage des courbes
lift_lda <- data.frame(
  "rpp" = lift_lda@x.values[[1]],
  "sensitivity" = lift_lda@y.values[[1]],
  "seuil" = lift_lda@alpha.values[[1]]
)
lift_cart <- data.frame(
  "rpp" = lift_cart@x.values[[1]],
  "sensitivity" = lift_cart@y.values[[1]],
  "seuil" = lift_cart@alpha.values[[1]]
)
lift_nb <- data.frame(
  "rpp" = lift_nb@x.values[[1]],
  "sensitivity" = lift_nb@y.values[[1]],
  "seuil" = lift_nb@alpha.values[[1]]
)

# Courbes LIFT des 3 modèles
ggplot(
  data = lift_lda,
  mapping = aes(x = rpp, y = sensitivity)
) +
  geom_line(
    mapping = aes(color = "LDA")
  ) +
  geom_line(
    data = lift_cart,
    mapping = aes(x = rpp, y = sensitivity, color = "CART")
  ) +
  geom_line(
    data = lift_nb,
    mapping = aes(x = rpp, y = sensitivity, color = "Naive Bayes")
  ) +
  geom_segment(
    mapping = aes(x = 0, y = 0, xend = 1, yend = 1),
    linetype = "dashed",
    size = .2
  ) +
  geom_segment(
    mapping = aes(x = 0, y = 0, xend = mean(spambase_test$is_spam == 1), yend = 1),
    linetype = "dashed",
    size = .2
  ) +
  geom_segment(
    mapping = aes(x = mean(spambase_test$is_spam == 1), y = 1, xend = 1, yend = 1),
    linetype = "dashed",
    size = .2
  ) +
  scale_x_continuous(labels = percent) +
  scale_y_continuous(labels = percent) +
  scale_colour_manual(
    values = c("CART" = "#1D88E5", "Naive Bayes" = "#FFC108", "LDA" = "#D81A60")
  ) +
  labs(
    x = "Taux de positifs",
    y = "Taux de vrais positifs",
    title = "Courbes LIFT",
    colour = "Modèle"
  ) +
  theme(
    legend.position = c(.8, .3)
  )


# Random forest -------------------------------------------------------------------------------

# Pour information, on teste une forêt aléatoire sur ces données

# Hyper-paramètres à tester (pour test)
grid <- expand.grid(.mtry = 5:6)

# définir la méthode de validation, ici 5-fold cross validation
control <- trainControl(method = "cv", number = 5)

# on optimise les paramètres du modèle
rf <- train(
  x = spambase_train %>% select(-is_spam), # prédicteurs
  y = spambase_train$is_spam, # réponse
  method = "rf", # classifieur utilisé, ici forêt aléatoire
  trControl = control, # méthode d'échantillonage, ici 5-fold CV
  tuneGrid = grid # liste des paramètres à comparer
)


prob_rf <- predict(rf, spambase_test, type = "prob")[, 2] %>% prediction(spambase_test$is_spam)

# generation des points pour construction de la courbe ROC
roc_rf <- performance(prob_rf, "tpr", "fpr")

# construction des data.frames nécessaires à l'affichage des coubres
roc_rf <- data.frame(
  "fpr" = roc_rf@x.values[[1]],
  "tpr" = roc_rf@y.values[[1]],
  "seuil" = roc_rf@alpha.values[[1]]
)

# courbes ROC des 3 modèles
ggplot(
  data = roc_lda,
  mapping = aes(x = fpr, y = tpr)
) +
  # courbe du modèle LDA
  geom_line(
    mapping = aes(color = "LDA")
  ) +
  # courbe du modèle cart
  geom_line(
    data = roc_cart,
    mapping = aes(x = fpr, y = tpr, color = "CART")
  ) +
  # courbe du modèle NB
  geom_line(
    data = roc_nb,
    mapping = aes(x = fpr, y = tpr, color = "Naive Bayes")
  ) +
  # courbe du modèle RF
  geom_line(
    data = roc_rf,
    mapping = aes(x = fpr, y = tpr, color = "RF")
  ) +
  # ajout du modèle aléatoire
  geom_segment(
    mapping = aes(x = 0, y = 0, xend = 1, yend = 1),
    linetype = "dashed",
    size = .2
  ) +
  # ajout du meilleur modèle
  geom_segment(
    mapping = aes(x = 0, y = 0, xend = 0, yend = 1),
    linetype = "dashed",
    size = .2
  ) +
  geom_segment(
    mapping = aes(x = 0, y = 1, xend = 1, yend = 1),
    linetype = "dashed",
    size = .2
  ) +
  # valeurs des axes en pourcentage
  scale_x_continuous(labels = percent) +
  scale_y_continuous(labels = percent) +
  # définition des couleurs des courbes
  scale_colour_manual(
    values = c("CART" = "#1D88E5", "Naive Bayes" = "#FFC108", "LDA" = "#D81A60", "RF" = "#228B22")
  ) +
  # définition des noms des axes, légende et titre
  labs(
    x = "Taux de faux positifs",
    y = "Taux de vrais positifs",
    title = "Courbes ROC",
    colour = "Modèle"
  ) +
  # position de la légende sur le graphe et non pas à côté
  theme(
    legend.position = c(.8, .3)
  )

# AUC
performance(prob_lda, "auc")@y.values[[1]]
performance(prob_cart, "auc")@y.values[[1]]
performance(prob_nb, "auc")@y.values[[1]]
performance(prob_rf, "auc")@y.values[[1]]

# generation des points pour construction de la courbe LIFT
lift_rf <- performance(prob_rf, "sens","rpp")

# construction des data.frames nécessaires à l'affichage des coubres
lift_rf <- data.frame(
  "rpp" = lift_rf@x.values[[1]],
  "sensitivity" = lift_rf@y.values[[1]],
  "seuil" = lift_rf@alpha.values[[1]]
)

# courbes LIFT des 3 modèles
ggplot(
  data = lift_lda,
  mapping = aes(x = rpp, y = sensitivity)
) +
  geom_line(
    mapping = aes(color = "LDA")
  ) +
  geom_line(
    data = lift_cart,
    mapping = aes(x = rpp, y = sensitivity, color = "CART")
  ) +
  geom_line(
    data = lift_nb,
    mapping = aes(x = rpp, y = sensitivity, color = "Naive Bayes")
  ) +
  geom_line(
    data = lift_rf,
    mapping = aes(x = rpp, y = sensitivity, color = "RF")
  ) +
  geom_segment(
    mapping = aes(x = 0, y = 0, xend = 1, yend = 1),
    linetype = "dashed",
    size = .2
  ) +
  geom_segment(
    mapping = aes(x = 0, y = 0, xend = mean(spambase_test$is_spam == 1), yend = 1),
    linetype = "dashed",
    size = .2
  ) +
  geom_segment(
    mapping = aes(x = mean(spambase_test$is_spam == 1), y = 1, xend = 1, yend = 1),
    linetype = "dashed",
    size = .2
  ) +
  scale_x_continuous(labels = percent) +
  scale_y_continuous(labels = percent) +
  scale_colour_manual(
    values = c("CART" = "#1D88E5", "Naive Bayes" = "#FFC108", "LDA" = "#D81A60", "RF" = "#228B22")
  ) +
  labs(
    x = "Taux de positifs",
    y = "Taux de vrais positifs",
    title = "Courbes LIFT",
    colour = "Modèle"
  ) +
  theme(
    legend.position = c(.8, .3)
  )
