def zero_one_loss(predictions: list, expected: list):
    total_loss = [p != e  for p, e in zip(predictions, expected)]
    return sum(total_loss)

def hinge_loss(predictions: list, probabilities: list):
    losses = [1 - probs[y_hat] for y_hat, probs in zip(predictions, probabilities)]
    return sum(losses)
