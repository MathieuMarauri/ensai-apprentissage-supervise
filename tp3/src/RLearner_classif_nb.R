
# Integrate Naive Bayes model from klaR package into mlr. More info here:
# https://mlr.mlr-org.com/articles/tutorial/create_learner.html

# step 1 - set up properties of the learner
makeRLearner.classif.nb <- function() {
  makeRLearnerClassif(
    cl = "classif.nb",
    package = "klaR",
    par.set = makeParamSet(
      makeNumericLearnerParam(id = "fL", default = 0, lower = 0),
      makeNumericLearnerParam(id = "adjust", default = 1, lower = 1),
      makeLogicalLearnerParam(id = "usekernel", default = FALSE)
    ),
    properties = c("twoclass", "multiclass", "numerics", "factors", "prob", "missings"),
    name = "Naive Bayes",
    short.name = "nb",
    note = "Priors are set to class proportions in the training set"
  )
}

# step 2 - create the train function
trainLearner.classif.nb <- function(.learner, .task, .subset, .weights = NULL, ...) {
  f <- getTaskFormula(.task)
  d <- getTaskData(.task, .subset)
  klaR::NaiveBayes(f, data = d, ...)
}

# step 3 - create the predict function
predictLearner.classif.nb <- function(.learner, .model, .newdata, ...) {
  type = switch(.learner$predict.type, prob = "prob", "class")
  p <- predict(.model$learner.model, newdata = .newdata, type = type, ...)
  if (.learner$predict.type == "response") {
    return(p$class)
  } else {
    return(p$posterior)
  }
}

# step 4 - register the functions
registerS3method("makeRLearner", "classif.nb", makeRLearner.classif.nb)
registerS3method("trainLearner", "classif.nb", trainLearner.classif.nb)
registerS3method("predictLearner", "classif.nb", predictLearner.classif.nb)
