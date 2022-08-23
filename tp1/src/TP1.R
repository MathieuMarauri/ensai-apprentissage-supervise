
# Correction du TP1

# Set-up ---------------------------------------------------------------------------------

library("car") # fonction Boxplot
library("caret") # ensemble de fonctions nécessaires au processus d'apprentissage supervisé
library("class") # modèle KNN
library("e1071") # librairie nécessaire pour la fonction tune.knn
library("FactoMineR") # ACM
library("FNN") # modèle KNN plus rapide
library("ggplot2") # visualisation de données
library("klaR") # modèle bayésien naïf
library("mlbench") # dataset Vehicle
library("skimr") # visualisation de l'ensemble d'un dataset

# Set default ggplot theme
theme_set(
  theme_light(
  base_size = 20
  ) +
  theme(
    text = element_text(family = "Gibson", colour = "gray10"),
    panel.border = element_blank(),
    axis.line = element_line(colour = "gray50", size = .5),
    axis.ticks = element_blank(),
    strip.background = element_rect(colour = "gray50", fill = "transparent", size = .7),
    strip.text.x = element_text(colour = "gray10"),
    strip.text.y = element_text(colour = "gray10"),
    legend.key.size = unit(1.5, "cm")
  )
)

# Set default scales
scale_colour_continuous <- function(...) ggplot2::scale_colour_viridis_c(..., option = "viridis")
scale_colour_discrete <- function(...) ggplot2::scale_colour_viridis_d(..., option = "viridis")
scale_fill_continuous <- function(...) ggplot2::scale_fill_viridis_c(..., option = "viridis")
scale_fill_discrete <- function(...) ggplot2::scale_fill_viridis_d(..., option = "viridis")


# Exercice 1 ----------------------------------------------------------------------------------

### Question 3

# Nombre d'éléments dans l'échantillon
n_train <- 1000

# Générer X selon une loi normale (fixer une seed pour avoir des résultats similaires à chaque
# fois)
set.seed(1)
X <- rnorm(n_train)

# Générer U selon une loi uniforme (fixer une seed pour avoir des résultats similaires à chaque
# fois)
set.seed(2)
U <- runif(n_train)

# Générer le vecteur réponse Y selon le cas 2, transformer en facteur pour la fonction
# tune.knn (utilisée dans la suite)
Y1 <- rep(0, n_train)
Y1[X <= 0 & U <= 0.2] <- 1
Y1[X > 0 & U > 0.4] <- 1
Y <- as.factor(Y1)

# Créer un data.frame avec X et Y
donnees <- data.frame(X, Y)

# Visualiser les données
ggplot(
  data = donnees,
  mapping = aes(x = X, fill = Y)
) +
  geom_density(
    alpha = .7
  ) +
  labs(
    x = "X",
    y = "Densité",
    title = "Densité de X selon Y",
    fill = "Y"
  )
# Boxplot(data = donnees, X ~ Y, main = "Données apprentissage et validation")

# Refaire la même chose pour la création de l'échantillon test
n_test <- 200
set.seed(3)
X <- rnorm(n_test)
set.seed(4)
U <- runif(n_test)
Y2 <- rep(0, n_test)
Y2[X <= 0 & U <= 0.2] <- 1
Y2[X > 0 & U > 0.4] <- 1
Y <- as.factor(Y2)
test <- data.frame(X, Y)
ggplot(
  data = test,
  mapping = aes(x = X, fill = Y)
) +
  geom_density(
    alpha = .7
  ) +
  labs(
    x = "X",
    y = "Densité",
    title = "Densité de X selon Y",
    fill = "Y"
  )
# Boxplot(data = test, X ~ Y, main = "Données apprentissage et validation")


### Question 4 : K plus proches voisins

# Un premier modèle
set.seed(123456)
knn_pred <- knn(
  train = as.matrix(donnees$X), # données d'apprentissage
  test = as.matrix(test$X), # données à prédire
  cl = donnees$Y, # vraies valeurs
  k = 10 # nombre de voisins
)

# Taux de bonnes prédictions
sum(knn_pred == test$Y) / n_test

# Taux d'erreur
1 - sum(knn_pred == test$Y) / n_test

# On cherche la meilleure valeur de K par cross validation
knn_cross_results <- tune.knn(
  x = donnees$X, # predicteurs
  y = donnees$Y, # réponse
  k = 1:50, # essayer knn avec K variant de 1 à 50
  tunecontrol = tune.control(sampling = "cross"), # utilisation de la cross validation
  cross = 5 # 5 blocks
)

# Visualiser les résultats pour chaque K
ggplot(
  data = knn_cross_results$performances,
  mapping = aes(x = k, y = error)
) +
  geom_line() +
  geom_point() +
  labs(
    x = "k",
    y = "Erreur de validation",
    title = "Evolution de l'erreur selon k"
  )
# summary(knn_cross_results)
# plot(knn_cross_results)

# On va ensuite tester la qualité du modèle avec le paramètre choisi
set.seed(123456)
knn_pred <- knn(
  train = as.matrix(donnees$X), # données d'apprentissage
  test = as.matrix(test$X), # données à prédire
  cl = donnees$Y, # vraies valeurs
  k = 30 # nombre de voisins
)

# Taux de bonnes prédictions
sum(knn_pred == test$Y) / n_test

# Taux d'erreur
1 - sum(knn_pred == test$Y) / n_test

# L'erreur est inférieur au risque de Bayes, notre estimation de l'espérance de l'erreur est
# peut-être mauvaise, utilisation d'un nouvel échantillon plus grand
n_test2 <- 2000
set.seed(6)
X <- rnorm(n_test2)
set.seed(7)
U <- runif(n_test2)
Y1 <- rep(0, n_test2)
Y1[X <= 0 & U <= 0.2] <- 1
Y1[X > 0 & U > 0.4] <- 1
Y <- as.factor(Y1)
test2 <- data.frame(X, Y)
set.seed(123456)
knn_pred <- knn(
  train = as.matrix(donnees$X), # données d'apprentissage
  test = as.matrix(test2$X), # données à prédire
  cl = donnees$Y, # vraies valeurs
  k = 30 # nombre de voisins
)
1 - sum(knn_pred == test2$Y) / n_test2 # c'est mieux


### Question 5 : Classifieur Bayésien naïf

# Un premier modèle
naive_bayes_model <- NaiveBayes(
  formula = Y ~ X,
  data = donnees
)

# Prédiction avec ce modèle sur les données de test
naive_bayes_pred <- predict(
  object = naive_bayes_model,
  newdata = test
)

# Calculer la précision et l'erreur
sum(naive_bayes_pred$class == test$Y) / n_test
1 - sum(naive_bayes_pred$class == test$Y) / n_test

# Définir la liste des hyper-paramètres possibles pour le classifieur bayésien naïf
hyperparam_grid <- expand.grid(
  usekernel = TRUE, # si vrai utilisation d'un noyau sinon gaussien
  adjust = seq(1, 5, by = 1) # largeur de bande
)
hyperparam_grid <- rbind(hyperparam_grid, data.frame(usekernel = FALSE, adjust = 1))

# Créer un échantillon de validation pour évaluer l'erreur sans biais
set.seed(123456)
train_index <- sample(1:nrow(donnees), size = nrow(donnees) * .8)
validation_index <- setdiff(1:nrow(donnees), train_index)
train <- donnees[train_index, ]
validation <- donnees[validation_index, ]

# Entraîner autant de modèles qu'il y a de combinaisons et renvoyer l'erreur
results <- cbind(hyperparam_grid, "error" = 0)
for (i in 1:nrow(results)) {
  naive_bayes_model <- NaiveBayes(
    formula = Y ~ X,
    data = train,
    usekernel = results$usekernel[i],
    adjust = results$adjust[i]
  )
  naive_bayes_pred <- predict(
    object = naive_bayes_model,
    newdata = validation
  )
  results[i, "error"] <- 1 - sum(naive_bayes_pred$class == validation$Y) / length(validation_index)
}

# Visualiser les résultats
ggplot(
  data = results,
  mapping = aes(x = adjust, y = error, color = usekernel)
) +
  geom_line() +
  geom_point() +
  labs(
    x = "Largeur de bande",
    y = "Erreur",
    title = "Comparaison des hyper-paramètres",
    color = "Utilisation d'un noyau ?"
  ) +
  theme(
    legend.position = c(.8, .22)
  )

# Le modèle optimal
naive_bayes_model <- NaiveBayes(
  formula = Y ~ X,
  data = donnees,
  usekernel = TRUE,
  fL = 0,
  adjust = 1
)

# Prédiction avec ce modèle sur les données de test
naive_bayes_pred <- predict(
  object = naive_bayes_model,
  newdata = test
)

# Calculer la précision et l'erreur
sum(naive_bayes_pred$class == test$Y) / n_test
1 - sum(naive_bayes_pred$class == test$Y) / n_test

# nettoyer la session
rm(list = setdiff(ls(), c("scale_colour_continuous", "scale_colour_discrete",
                          "scale_fill_continuous", "scale_fill_discrete")))


# Exercice 2 ----------------------------------------------------------------------------------

### Question 1 : lecture de la table

# Importer le dataset vehicule et avoir un aperçu des données
data(Vehicle)
summary(Vehicle)
# skim(Vehicle)

# On décompose la base en un échantillon d'apprentissage et un échantillon de test
set.seed(123456)
train_index <- sample(1:nrow(Vehicle), size = nrow(Vehicle) * .8)
test_index <- setdiff(1:nrow(Vehicle), train_index)
train <- Vehicle[train_index, ]
test <- Vehicle[test_index, ]

# Donner la liste des prédicteurs
predicteurs <- names(Vehicle)[-19]
# on peut sélectionner moins de variable en faisant una analyse bivariée, une ACP pour garder
# seulement les premiers axes, ...

### Question 3 : K plus proches voisins

# Mise à niveau des variables
train_scaled <- scale(train[, predicteurs])
test_scaled <- scale(
  x = test[, predicteurs],
  center = attributes(train_scaled)[["scaled:center"]],
  scale = attributes(train_scaled)[["scaled:scale"]]
)

# relancer la cross validation plusieurs fois pour déterminer le meilleur K
best_k <- numeric(length = 20)
for (i in 1:20) {
  knn_cross_ressults <- tune.knn(
    x = train_scaled,
    y = train$Class,
    k = 1:50,
    tunecontrol = tune.control(sampling = "cross"),
    cross = 5
  )
  best_k[i] <- knn_cross_ressults$best.parameters[[1]]
  print(sprintf("Itération %s : meilleur K = %s", i, knn_cross_ressults$best.parameters))
}
table(best_k)

# On va ensuite tester la qualité du modèle avec le paramètre choisi
set.seed(123456)
knn_pred <- knn(
  train = train_scaled, # données d'apprentissage
  test = as.matrix(scale(test[, predicteurs])), # données à prédire
  cl = train$Class, # vraies valeurs
  k = 3 # nombre de voisins
)

# Taux de bonnes prédictions
sum(knn_pred == test$Class) / length(test_index)

# Taux d'erreur
1 - sum(knn_pred == test$Class) / length(test_index)


### Question 4 : Classifieur Bayésien naïf

# définir la liste des hyper-paramètres possibles pour le classifieur bayésien naïf
hyperparam_grid <- expand.grid(
  usekernel = TRUE, # si vrai utilisation d'un noyau sinon gaussien
  fL = 0, # correction avec lissage de Laplace (ici ce paramètre n'est pas nécessaire, x étant continue)
  adjust = seq(1, 5, by = 1) # largeur de bande
)
hyperparam_grid <- rbind(hyperparam_grid, data.frame(usekernel = FALSE, fL = 0, adjust = 1))

# définir la méthode de validation, ici 10-fold cross validation
control <- trainControl(method = "cv", number = 5)
# control <- trainControl(method = "LGOCV", p = 0.8, number = 1)

# définir les entrées de al fonction train de caret, nommer les éléments est une contrainte de caret
x <- train[, predicteurs]
y <- train$Class

# On optimise les paramètres du modèle
naive_bayes <- train(
  x = x, # prédicteurs
  y = y, # réponse
  # preProc = c("BoxCox", "center", "scale", "pca"), # utilisation de différents pré-traitement
  method = "nb", # classifieur utilisé, ici Naive Bayes
  trControl = control, # méthode d'échantillonnage, ici 5-fold CV
  tuneGrid = hyperparam_grid # liste des paramètres à comparer
)

# Visualisation des résultats
ggplot(
  data = naive_bayes$results,
  mapping = aes(x = adjust, y = Accuracy, color = usekernel)
) +
  geom_line() +
  geom_point() +
  labs(
    x = "Largeur de bande",
    y = "Précision",
    title = "Comparaison des hyper-paramètres",
    color = "Utilisation d'un noyau ?"
  ) +
  theme(
    legend.position = c(.8, .8)
  )
# plot(naive_bayes)

# On va ensuite tester la qualité du modèle avec le paramètre choisi (modèle non paramétrique
# et BW = 1)
naive_bayes_pred <- predict(naive_bayes, test)
naive_bayes_pred_prob <- predict(naive_bayes, test, type = "prob")
# Les warnings sont dus à la présence d'outlier dans les données de test.

# calcule de l'erreur
sum(naive_bayes_pred == test[, "Class"]) / length(test_index)
1 - sum(naive_bayes_pred == test[, "Class"]) / length(test_index)

# nettoyer la session
rm(list = setdiff(ls(), c("scale_colour_continuous", "scale_colour_discrete",
                          "scale_fill_continuous", "scale_fill_discrete")))


# Exercice 3 ----------------------------------------------------------------------------------

### Lecture de la table

# Importer le dataset airbnb et avoir un aperçu des données
airbnb <- read.csv(file = "tp1/data/AB_NYC_2019.csv")
summary(airbnb)
# skim(airbnb)

# On décompose la base en un échantillon d'apprentissage et un échantillon de test
set.seed(123456)
train_index <- sample(1:nrow(airbnb), size = nrow(airbnb) * .7)
test_index <- setdiff(1:nrow(airbnb), train_index)
train <- airbnb[train_index, ]
test <- airbnb[test_index, ]

### Question 3 : K plus proches voisins

#### Variables continues uniquement

# On cherche la meilleure valeur de K par cross validation
knn_fit <- train(
  x = train[, c("latitude", "longitude")], # prédicteurs
  y = train$price, # réponse
  tuneGrid = data.frame(k = seq(100, 200, by = 5)), # nombre de voisins
  method = "knn", # knn classifieur
  trControl = trainControl(method = "cv", number = 5) # 5-fold CV
)

# Visualisation des résultats
ggplot(
  data = knn_fit$results,
  mapping = aes(x = k, y = RMSE)
) +
  geom_line() +
  geom_point() +
  labs(
    x = "Valeurs de k",
    y = "Erreur quadratique",
    title = "Evaluation de l'erreur sur données de validation"
  )
# plot(knn_fit)

# Effectuer une prédiction sur les données test
knn_pred <- predict(
  object = knn_fit,
  newdata = test
)

# Calcul de l'erreur
sqrt(mean((knn_pred - test$price)^2))

#### Encoder les variables catégorielles

# Transformer les variables catégorielles en factor et récupérer la liste des levels pour
# l'appliquer au set de test
train$room_type <- as.numeric(
  factor(
    x = train$room_type,
    levels = unique(airbnb$room_type)
  )
)
train$neighbourhood_group <- as.numeric(
  factor(
    x = train$neighbourhood_group,
    levels = unique(airbnb$neighbourhood_group)
  )
)
train$neighbourhood <- as.numeric(
  factor(
    x = train$neighbourhood,
    levels = unique(airbnb$neighbourhood)
  )
)

# Sur les données de test
test$room_type <- as.numeric(
  factor(
    x = test$room_type,
    levels = unique(airbnb$room_type)
  )
)
test$neighbourhood_group <- as.numeric(
  factor(
    x = test$neighbourhood_group,
    levels = unique(airbnb$neighbourhood_group)
  )
)
test$neighbourhood <- as.numeric(
  factor(
    x = test$neighbourhood,
    levels = unique(airbnb$neighbourhood)
  )
)

# Mise à niveau des variables
train_scaled <- scale(train[, c("latitude", "longitude", "room_type", "neighbourhood_group", "neighbourhood")])
test_scaled <- scale(
  x = test[, c("latitude", "longitude", "room_type", "neighbourhood_group", "neighbourhood")],
  center = attributes(train_scaled)[["scaled:center"]],
  scale = attributes(train_scaled)[["scaled:scale"]]
)

# On cherche la meilleure valeur de K par cross validation
knn_fit <- train(
  x = train_scaled, # prédicteurs
  y = train$price, # réponse
  tuneGrid = data.frame(k = seq(40, 100, by = 5)), # nombre de voisins
  method = "knn", # knn classifieur
  trControl = trainControl(method = "cv", number = 5) # 5-fold CV
)

# visualisation des résultats
ggplot(
  data = knn_fit$results,
  mapping = aes(x = k, y = RMSE)
) +
  geom_line() +
  geom_point() +
  labs(
    x = "Valeurs de k",
    y = "Erreur quadratique",
    title = "Evaluation de l'erreur sur données de validation"
  )
# plot(knn_fit)

# Effectuer une prédiction sur les données test
knn_pred <- predict(
  object = knn_fit,
  newdata = test_scaled
)

# Calcul de l'erreur
sqrt(mean((knn_pred - test$price)^2))


#### ACM

# Reconstruction des objets train et test
set.seed(123456)
train_index <- sample(1:nrow(airbnb), size = nrow(airbnb) * .7)
test_index <- setdiff(1:nrow(airbnb), train_index)
train <- airbnb[train_index, ]
test <- airbnb[test_index, ]

# Effectuer l'acm sur les données de train
acm <- MCA(
  X = train[, c("neighbourhood_group","room_type")],
  ncp = 20,
  graph = FALSE
)

# Mise à niveau des variables continues
train_scaled <- scale(train[, c("latitude", "longitude")])
test_scaled <- scale(
  x = test[, c("latitude", "longitude")],
  center = attributes(train_scaled)[["scaled:center"]],
  scale = attributes(train_scaled)[["scaled:scale"]]
)

# Ajouter les coordonnées aux données continues
train_acm <- cbind(train_scaled, acm$ind$coord)

# Entraîner le modèle de knn
knn_fit <- train(
  x = train_acm, # prédicteurs
  y = train$price, # réponse
  tuneGrid = data.frame(k = seq(40, 300, by = 20)), # nombre de voisins
  method = "knn", # knn classifieur
  trControl = trainControl(method = "cv", number = 5) # 5-fold CV
)

# visualisation des résultats
ggplot(
  data = knn_fit$results,
  mapping = aes(x = k, y = RMSE)
) +
  geom_line() +
  geom_point() +
  labs(
    x = "Valeurs de k",
    y = "Erreur quadratique",
    title = "Evaluation de l'erreur sur données de validation"
  )
# plot(knn_fit)

# Préparer les données test pour faire une prédiction
acm_pred <- predict.MCA(
  object = acm,
  newdata = test[, c("neighbourhood_group","room_type")]
)

# Effectuer une prédiction sur les données test
knn_pred <- predict(
  object = knn_fit,
  newdata = cbind(test_scaled, acm_pred$coord)
)

# Calcul de l'erreur
sqrt(mean((knn_pred - test$price)^2))
