
# Set-up ---------------------------------------------------------------------------------

library("data.table") # Fast dataset manipulation
library("ggplot2") # Data visualisations using the Grammar of Graphics
library("mlr3")
library("mlr3learners")
library("mlr3tuning")
library("mlr3viz")
library("paradox")
# library("precrec")
library("scales")

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


# Création de la task ------------------------------------------------------------------------

# Importer la table spambase dans l'environnement
spambase <- fread('tp3/data/spambase.csv')

# Transformation de la variable cible en factor
spambase$is_spam <- as.factor(spambase$is_spam)

# Enlever la colonne V1
spambase$V1 <- NULL

# Création des indices pour les datasets d'apprentissage et de test
set.seed(123)
train_ids <- sample(seq_len(nrow(spambase)), 0.8 * nrow(spambase))
test_ids <- setdiff(seq_len(nrow(spambase)), train_ids)

# Définir une classification task
task <- TaskClassif$new(
  id = "spambase",
  backend = spambase[train_ids,],
  target = "is_spam",
  positive = "1"
)


# Un premier modèle --------------------------------------------------------------------------

# Définir le modèle
cart <- lrn("classif.rpart", predict_type = "prob")

# Entraîner le modèle (avec ses hyper-paramètres par défauts)
cart$train(task)

# Effectuer des prédictions sur le test set
prediction <- cart$predict_newdata(task, newdata = spambase[test_ids,])

prediction$confusion

# Métriques d'évaluation
measures <- list(
  msr("classif.acc"),
  msr("classif.auc"),
  msr('classif.tpr'),
  msr('classif.tnr')
)

# Performance du modèle
prediction$score(measures)

# Courbe ROC
autoplot(prediction, type = "roc") +
  labs(
    title = 'Courbe ROC'
  )

# Extraction des données pour la courbe lift
lift_data = lapply(
  X = seq(from = 0, to = 1, by = 0.01),
  FUN = function(threshold) {
    data.table(
      tpr = prediction$set_threshold(threshold)$score(msr('classif.tpr')),
      tp = prediction$set_threshold(threshold)$score(msr('classif.tp')),
      fp = prediction$set_threshold(threshold)$score(msr('classif.fp'))
    )
  }
)
lift_data = rbindlist(lift_data)
lift_data[, pp := (tp + fp) / length(test_ids)]

# Courbe LIFT
ggplot(
  data = lift_data,
  mapping = aes(x = pp, y = tpr)
) +
  geom_line() +
  geom_segment(
    mapping = aes(x = 0, y = 0, xend = 1, yend = 1),
    linetype = "dashed",
    size = .2
  ) +
  geom_segment(
    mapping = aes(x = 0, y = 0, xend = mean(spambase[test_ids,]$is_spam == 1), yend = 1),
    linetype = "dashed",
    size = .2
  ) +
  geom_segment(
    mapping = aes(x = mean(spambase[test_ids,]$is_spam == 1), y = 1, xend = 1, yend = 1),
    linetype = "dashed",
    size = .2
  ) +
  scale_x_continuous(labels = percent) +
  scale_y_continuous(labels = percent) +
  labs(
    x = "Taux de positifs",
    y = "Taux de vrais positifs",
    title = "Courbe LIFT"
  )


# Optimisation des hyper-paramètres ----------------------------------------------------------

# Lister les hyper-paramètres
cart$param_set

# Définir la plage des valeurs à tester
search_space = ps(
  cp = p_dbl(lower = 0.001, upper = 0.1),
  minsplit = p_int(lower = 1, upper = 20)
)

# Optimiser les hyper-paramètres
hyperparam_optim <- TuningInstanceSingleCrit$new(
  task = task,
  learner = cart,
  resampling = rsmp("cv", folds = 5),
  measure = msr("classif.acc"),
  search_space = search_space,
  # terminator = trm("clock_time", stop_time = Sys.time() + 120)
  terminator = trm("evals", n_evals = 1000)
)
tuner <- tnr("grid_search", param_resolutions = c(cp = 100, minsplit = 10))
tuner$optimize(hyperparam_optim)

# Utiliser les hyper-paramètres optimaux dans le modèle
cart$param_set$values <- hyperparam_optim$result_learner_param_vals

# Entraîner le modèle (avec les hyper-paramètres choisis)
cart$train(task)

# Effectuer des prédictions sur le test set
prediction <- cart$predict_newdata(task, newdata = spambase[test_ids,])

# Matrice de confusion
prediction$confusion

# Performance du modèle
prediction$score(list(
  msr("classif.acc"),
  msr("classif.auc")
))


# Comparaison de modèles ---------------------------------------------------------------------

# CART
# Définir le modèle cart
cart <- lrn("classif.rpart", predict_type = "prob")

# Lister les hyper-paramètres de cart
cart$param_set

# Définir les hyper-paramètres à tester pour le modèle cart
search_space_cart <- ps(
  cp = p_dbl(lower = 0.001, upper = 0.1),
  minsplit = p_int(lower = 1, upper = 20)
)

# Optimisation des hyper-paramètres
cart_tuned <- AutoTuner$new(
  learner = cart,
  resampling = rsmp("cv", folds = 5),
  measure = msr("classif.ce"),
  search_space = search_space_cart,
  terminator = trm("evals", n_evals = 100),
  tuner = tnr("grid_search", param_resolutions = c(cp = 100, minsplit = 10))
)

# KNN
# Définir le modèle knn
knn <- lrn("classif.kknn", predict_type = "prob")

# Lister les hyper-paramètres de knn
knn$param_set

# Utiliser un kernel rectangulaire pour faire un modèle sans poids
knn$param_set$values$kernel <- "rectangular"

# Définir les hyper-paramètres à tester pour le modèle knn
search_space_knn <- ps(
  k = p_int(lower = 1, upper = 200)
)

# Optimisation des hyper-paramètres
knn_tuned <- AutoTuner$new(
  learner = knn,
  resampling = rsmp("cv", folds = 5),
  measure = msr("classif.ce"),
  search_space = search_space_knn,
  terminator = trm("evals", n_evals = 200),
  tuner = tnr("grid_search", resolution = 200)
)

# Naive Bayes
# Définir le modèle bayésien naïf
nb <- lrn("classif.naive_bayes", predict_type = "prob")

# Lister les hyper-paramètres du modèle bayésien naïf
nb$param_set

# Définir les hyper-paramètres à tester pour le modèle bayésien naïf
search_space_nb <- ps(
  laplace = p_int(lower = 0, upper = 5)
)

# Optimisation des hyper-paramètres
nb_tuned <- AutoTuner$new(
  learner = nb,
  resampling = rsmp("cv", folds = 5),
  measure = msr("classif.ce"),
  search_space = search_space_nb,
  terminator = trm("evals", n_evals = 6),
  tuner = tnr("grid_search", resolution = 6)
)

# Analyse discriminante linéaire
# Définir le modèle d'analyse discriminante linéaire
lda <- lrn("classif.lda", predict_type = "prob")

# Random Forest
# Définir le modèle de forêt
rf <- lrn("classif.ranger", predict_type = "prob")

# Lister les hyper-paramètres du modèle de forêt
rf$param_set

# Définir les hyper-paramètres à tester pour le modèle de forêt
search_space_rf <- ps(num.trees = p_int(lower = 50, upper = 200))

# Optimisation des hyper-paramètres
rf_tuned <- AutoTuner$new(
  learner = rf,
  resampling = rsmp("cv", folds = 5),
  measure = msr("classif.ce"),
  search_space = search_space_rf,
  terminator = trm("evals", n_evals = 15),
  tuner = tnr("grid_search", resolution = 15)
)

# Comparaison des différents modèles
design <- benchmark_grid(
  tasks = task,
  learners = list(
    cart_tuned,
    knn_tuned,
    nb_tuned,
    lda,
    rf_tuned
  ),
  resamplings = rsmp("holdout", ratio = .8)
)
bmr <- benchmark(
  design = design,
  store_models = TRUE
)

# Sauvegarde du résultat
saveRDS(bmr, 'tp3/data/benchmark_result.rds')
bmr <- readRDS('tp3/data/benchmark_result.rds')

# Comparaison des modèles selon différentes métriques de performance
results <- bmr$aggregate(list(
  msr('classif.acc'),
  msr("classif.auc")
))

# Courbes ROC
autoplot(bmr, type = 'roc')  +
  labs(
    title = 'Courbe ROC'
  )

# Sélection du meilleur modèle
final_model <- bmr$score(msr('classif.acc'))[classif.acc == max(classif.acc)]$learner[[1]]

# Hyper-paramètres choisis
final_model$learner$param_set

