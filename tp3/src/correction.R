

# Setup ---------------------------------------------------------------------------------------

# Packages à installer avant le TP (utiliser install.packahges(""))
library('rpart')
library('rpart.plot')
library('klaR')
library('ROCR')


# Données -------------------------------------------------------------------------------------

# Charger les données dans l'environnement de travail et construire un échantillon d'apprentissage et de test

# Lecture des données
spambase <- read.csv(
  "tp3/data/spambase.csv", # Chemin vers le fichier à importer
  row.names = 1 # La première colonne est le nom des lignes
)

# Transformer la variable cible en factor
spambase$is_spam <- as.factor(spambase$is_spam)

# Création des 2 échantillons de données
set.seed(1234)
test_rows <- sample.int(nrow(spambase), nrow(spambase)/4)
test <- spambase[test_rows,]
base <- spambase[-test_rows,]


# CART ----------------------------------------------------------------------------------------

# Construction d'un arbre CART
model_cart <- rpart(
  formula = is_spam ~ ., # Quelle variable à prédire en fonction de quelles autres, . pour spécifier toutes les autres variables
  data = base, # Les données d'apprentissage sur lesquelles le modèle sera construit
  method = "class" # Indiquer au modèle qu'on veut faire une classification et pas une régression
)
rpart.plot(model_cart)

# Effectuer des prédictions sur les données de test.
# Prédire les classes (spam ou non) pour ensuite comparer avec les vraies valeurs et mesurer l'erreur
pred_cart <- predict(
  object = model_cart, # Le modèle entraîné à l'étape précédente
  newdata = test, # Les nouvelles données sur lesquelles effectuer les prédictions, ici les données de test
  type = "class" # Le type de prédiction, ici les classes directement (spam ou non spam)
)

# Matrice de confusion avec en ligne les vraies valeurs et en colonnes les valeurs prédites
table(test$is_spam, pred_cart)

# Calcul du taux d'erreur en test = nombre de prédictions différentes de la réalité / nombre de prédictions.
sum(pred_cart != test$is_spam) / nrow(test)


# Bayésien naïf -------------------------------------------------------------------------------

# Construction d'un modèle bayésien naïf.
# Si on souhaite utiliser le modèle bayésien naïf sur toutes les variables, le modèle renverra une erreur.
# En effet une variable ne présente aucune variabilité conditionnellement à la variable cible, elle est retirée pour effectuer le modèle.
model_nb <- NaiveBayes(
  formula = is_spam ~ ., # Quelle variable à prédire en fonction de quelles autres, . pour spécifier toutes les autres variables
  data = base[, names(base) != 'word_freq_857'] # Les données d'apprentissage sur lesquelles le modèle sera construit
)

# Effectuer des prédictions sur les données de test.
# Prédire les classes (spam ou non) pour ensuite comparer avec les vraies valeurs et mesurer l'erreur
pred_nb <- predict(
  object = model_nb, # Le modèle entraîné à l'étape précédente
  newdata = test[, names(test) != 'word_freq_857'] # Les nouvelles données sur lesquelles effectuer les prédictions, ici les données de test
)
# La fonction predict utilisée sur un modèle naive bayes renvoie des informations différentes, pour obtenir les classes prédites il faut récupérer le bon objet.
pred_nb <- pred_nb$class

# Matrice de confusion avec en ligne les vraies valeurs et en colonnes les valeurs prédites
table(test$is_spam, pred_nb)

# Calcul du taux d'erreur en test = nombre de prédictions différentes de la réalité / nombre de prédictions.
sum(pred_nb != test$is_spam) / nrow(test)


# LDA -----------------------------------------------------------------------------------------

# Construction d'un modèle d'analyse discriminante linéaire
model_lda <- lda(
  formula = is_spam ~ .,
  data = base
)

# Effectuer des prédictions sur les données test
pred_lda <- pred_nb <- predict(
  object = model_lda, # Le modèle entraîné à l'étape précédente
  newdata = test # Les nouvelles données sur lesquelles effectuer les prédictions, ici les données de test
)
# La fonction predict utilisée sur un modèle lda renvoie des informations différentes, pour obtenir les classes prédites il faut récupérer le bon objet.
pred_lda <- pred_lda$class

# Matrice de confusion avec en ligne les vraies valeurs et en colonnes les valeurs prédites
table(test$is_spam, pred_lda)

# Calcul du taux d'erreur en test = nombre de prédictions différentes de la réalité / nombre de prédictions.
sum(pred_lda != test$is_spam) / nrow(test)


# Courbes ROC ---------------------------------------------------------------------------------

# Pour obtenir la courbes ROC il faut construire les vecteurs de TVP et TFP pour chaque modèle.
# On commence par effectuer les prédictions sur les données test en conservant les probabilités d'être de la classe 1.
# Ensuite on utilise la fonction prediction du package ROCR pour calculer les matrices de confusion pour différents seuils correspondants aux différrentes probabiltés prédites par le modèle.
# La fonction performance permet ensuite de récupérer les indicateurs nécessaires à la construction de la courbe ROC : TVP et TFP

# Prédiction des probabilités pour chaque individu de test d'être de la classe 1
pred_cart <- predict(object = model_cart, newdata = test, type = "prob")[,2]

# Construction de toutes les matrices de confusion pour tous les seuils
confmats_cart <- prediction(pred_cart, test$is_spam)

# Extraction des taux de vrais positifs et taux de faux positifs
roc_cart <- performance(confmats_cart, "tpr", "fpr")

# Extraction de l'aire sous la courbe
auc_cart <- performance(confmats_cart, "auc")

# Même chose pour le modèle bayésien naïf
pred_nb <- predict(object = model_nb, newdata = test[, names(test) != 'word_freq_857'])$posterior[,2]
confmats_nb <- prediction(pred_nb, test$is_spam)
roc_nb <- performance(confmats_nb, "tpr", "fpr")
auc_nb <- performance(confmats_nb, "auc")

# Même chose pour le modèle lda
pred_lda <- predict(object = model_lda, newdata = test)$posterior[,2]
confmats_lda <- prediction(pred_lda, test$is_spam)
roc_lda <- performance(confmats_lda, "tpr", "fpr")
auc_lda <- performance(confmats_lda, "auc")

# Tracer les courbes
plot(roc_cart, col = 1, lty = 1, lwd = 1, main = "Courbes ROC")
plot(roc_nb, col = 2, lty = 1, lwd = 1, add = TRUE)
plot(roc_lda, col = 3, lty = 1, lwd = 1, add = TRUE)
legend(
  "bottomright",
  cex = 0.6,
  legend = c(
   paste("CART AUC = ", round(as.numeric(auc_cart@y.values), 2)),
   paste("NB AUC = ", round(as.numeric(auc_nb@y.values), 2)),
   paste("LDA AUC = ", round(as.numeric(auc_lda@y.values), 2))
  ),
  col = 1:3, lty = 1:3, lwd = c(1, 3)
)


# Courbes LIFT --------------------------------------------------------------------------------

# Le même fonctionnement que précédemment est utilisé avec extraction des TVP et TP.

# Indicateurs LIFT pour CART
pred_cart <- predict(object = model_cart, newdata = test, type = 'prob')[,2]
confmats_cart <- prediction(pred_cart, test$is_spam)
lift_cart <- performance(confmats_cart, "sens", "rpp")

# Indicateurs LIFT pour le modèle bayésien naïf
pred_nb <- predict(object = model_nb, newdata = test[, names(test) != 'word_freq_857'])$posterior[,2]
confmats_nb <- prediction(pred_nb, test$is_spam)
lift_nb <- performance(confmats_nb, "sens", "rpp")

# Indicateurs LIFT pour le modèle lda
pred_lda <- predict(object = model_lda, newdata = test)$posterior[,2]
confmats_lda <- prediction(pred_lda, test$is_spam)
lift_lda <- performance(confmats_lda, "sens", "rpp")

# Tracer les courbes
plot(lift_cart, col = 1, lty = 1, lwd = 1, main = "Courbes LIFT")
plot(lift_nb, col = 2, lty = 1, lwd = 1, add = TRUE)
plot(lift_lda, col = 3, lty = 1, lwd = 1, add = TRUE)
legend(
  "bottomright",
  cex = 0.6,
  legend = c(
    paste("CART"),
    paste("NB"),
    paste("LDA")
  ),
  col = 1:3, lty = 1:3, lwd = c(1, 3)
)


