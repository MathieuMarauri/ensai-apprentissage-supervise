set.seed(111)


###################################################################
# Utiliser la table des données spambase de la librairie nutshell #
###################################################################
# library(nutshell)
# data(spambase)
### ou bien si probleme de version de package importer la base ###
spambase <- read.csv("tp3/data/spambase.csv", row.names = 1)

summary(spambase)

# on  crée deux échantillons
test_rows <- sample.int(nrow(spambase), nrow(spambase) / 4)
test <- spambase[test_rows, ]
base <- spambase[-test_rows, ]

dim(base)
dim(test)

##########################################################################
# Construire l'arbre CART puis élagage avec validation croisée
# pour leave one out on prendrait (nrow(base)-1) mais c'est long #
##########################################################################
# chargement de la librairie Rpart
library(rpart)

# pour de plus beaux graphiques : install.packages("rpart.plot")
library(rpart.plot)

CART <- rpart(
  is_spam ~ .,
  data = base,
  method = "class",
  parms = list(split = "gini")
)
summary(CART)
#  modalités de spam dans l'ordre alphabétique)
predCART <- predict(CART, test, type = "class") # donne pour chaque observation la modalité prédite

# Matrice de confusion avec en ligne les vraies valeurs et en colonnes les valeurs prédites
matCART <- table(test$is_spam, predCART)

# calcul du taux d'erreur en test
taux_err_CART <- sum(predCART != test$is_spam) / nrow(test)



#############################################################
# Analyse discriminante
#############################################################
library(MASS)
# Deux methodes d'analyse discriminante linéaire et quadratique
# erreur par validation croisée
# et erreur sur echantillon test

# analyse discriminante linéaire
lda(is_spam ~ ., data = base, CV = T)
disl <- lda(is_spam ~ ., data = base)
preddisl <- predict(disl, test)


####################################
# courbes ROC, AUC, courbes LIFT
####################################

library(ROCR)

ROCCART <- predict(CART, newdata = test, type = "prob")[, 2]
predCART <- prediction(ROCCART, test$is_spam)
perfCART <- performance(predCART, "tpr", "fpr") # pour courbe ROC
auc.tmp <- performance(predCART, "auc") # pour AUC
aucCART <- round(as.numeric(auc.tmp@y.values), 2)
LiftCART <- performance(predCART, "sens", "rpp") # pour courbe LIFT


ROClda <- predict(disl, test)$posterior[, 2]
predlda <- prediction(ROClda, test$is_spam)
perflda <- performance(predlda, "tpr", "fpr")
auc.tmp <- performance(predlda, "auc") # pour AUC
auclda <- round(as.numeric(auc.tmp@y.values), 2)
Liftlda <- performance(predlda, "sens", "rpp")


# tracer les courbes ROC en les superposant
# pour mieux comparer
plot(perfCART,
  col = 1, lty = 1, lwd = 1,
  main = "Courbes ROC"
)
plot(perflda, col = 2, lty = 2, lwd = 3, add = TRUE)
legend("bottomright",
  cex = 0.6,
  legend = c(
    paste("CART AUC= ", aucCART),
    paste("lda AUC= ", auclda)
  ),
  col = 1:2, lty = 1:2, lwd = c(1, 3)
)


# tracer les courbes LIFT en les superposant
# pour mieux comparer
plot(LiftCART, col = 1, lty = 1, lwd = 1, main = "Courbes LIFT")
plot(Liftlda, col = 2, lty = 2, lwd = 3, add = TRUE)
legend("bottomright",
  legend = c("CART", "lda"),
  cex = 0.6,
  col = 1:2,
  lty = 1:2, lwd = c(1, 3)
)

