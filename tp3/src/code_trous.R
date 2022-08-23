# Environnement de travail
setwd("???")

###################################################################
# Utiliser la table des données spambase de la librairie nutshell #
###################################################################
#library(nutshell)
#data(spambase)
### ou bien si probleme de version de package importer la base ###
spambase <- read.csv("spambase.csv", row.names=1)

library(caret)

summary(spambase)

#on  crée deux échantillons
test_rows = sample.int(nrow(spambase), nrow(spambase)/4)
test = spambase[test_rows,]
base = spambase[-test_rows,]

dim(base)
dim(test)

##########################################################################
# Construire l'arbre CART puis élagage avec validation croisée
# pour leave one out on prendrait (nrow(base)-1) mais c'est long #
##########################################################################
#chargement de la librairie Rpart
library(rpart)

#pour de plus beaux graphiques : install.packages("rpart.plot")
library(rpart.plot)

CART=rpart( ???~???,data=???, method="???",
            parms=list( split='gini'))
summary(CART)
#  modalités de spam dans l'ordre alphabétique)
predCART=predict(???,???,type="???") # donne pour chaque observation la modalité prédite

# Matrice de confusion avec en ligne les vraies valeurs et en colonnes les valeurs prédites
matCART=table(???$???,???)

# calcul du taux d'erreur en test
taux_err_CART= sum(???!= ???$???)/nrow(???)



#############################################################
# Analyse discriminante
#############################################################
library(MASS)
#Deux methodes d'analyse discriminante linéaire et quadratique
# erreur par validation croisée
# et erreur sur echantillon test

# analyse discriminante linéaire
lda(???~???,data=???,CV=???)
disl = lda(is_spam~.,data=???)
preddisl=predict(???,???)


####################################
# courbes ROC, AUC, courbes LIFT
####################################

library(ROCR)

ROCCART=predict(???,newdata=???,type="prob")[,2]
predCART=prediction(ROCCART,???$???)
perfCART=performance(???,"tpr","fpr") #pour courbe ROC
auc.tmp=performance(???,"auc") #pour AUC
aucCART= round(as.numeric(auc.tmp@y.values),2)
LiftCART=performance(???,"sens","rpp")#pour courbe LIFT


ROClda=predict(???,???)$posterior[,2]
predlda=prediction(???,???$???)
perflda=performance(???,"tpr","fpr")
auc.tmp=performance(???,"auc") #pour AUC
auclda= round(as.numeric(auc.tmp@y.values),2)
Liftlda=performance(???,"sens","rpp")


# tracer les courbes ROC en les superposant
# pour mieux comparer
plot(???,col=1,lty=1,lwd=1,
     main="Courbes ROC")
plot(???,col=2,lty=2,lwd=3,add=TRUE)
legend("bottomright",cex=0.6,
       legend=c(paste("CART AUC= ",???),
                paste("lda AUC= ",???)),
       col=1:2,lty=1:2, lwd=c(1,3))

#?legend

# tracer les courbes LIFT en les superposant
# pour mieux comparer
plot(???,col=1,lty=1,lwd=1,main="Courbes LIFT")
plot(???,col=2,lty=2,lwd=3,add=TRUE)
legend("bottomright",legend=c("???","???"),
       cex=0.6,
       col=1:2,
       lty=1:2, lwd=c(1,3))

#?plot.performance