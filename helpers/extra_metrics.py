def explained_variance_score(y_true, y_pred):
    return 1 - K.var(y_true-y_pred, axis=0)/K.var(y_true, axis=0)


def r2_score(y_true, y_pred):
    return 1 - K.sum(K.pow(y_true-y_pred, 2))/K.sum(K.pow(y_true-K.mean(y_true, keepdims=True), 2))
