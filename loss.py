import paddle.fluid as fluid
def mse_loss(y_true, y_pred):
    if y_true == 1:
        y_true = fluid.layers.ones_like(y_pred)
    else:
        y_true = fluid.layers.zeros_like(y_pred)
    return fluid.layers.mse_loss(y_pred, y_true)

def bce_loss(y_true, y_pred):
    if y_true == 1:
        y_true = fluid.layers.ones_like(y_pred)
    else:
        y_true = fluid.layers.zeros_like(y_pred)
    return fluid.layers.reduce_mean(fluid.layers.sigmoid_cross_entropy_with_logits(y_pred, y_true))
