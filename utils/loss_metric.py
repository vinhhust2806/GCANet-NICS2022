from keras import backend as K
import tensorflow as tf

smooth=1

def dice_coef(y_true, y_pred):
    y_truef=K.flatten(y_true)
    y_predf=K.flatten(y_pred)
    And=K.sum(y_truef* y_predf)
    return((2* And + smooth) / (K.sum(y_truef) + K.sum(y_predf) + smooth))
def dice_loss(y_true,y_pred):
  return 1-dice_coef(y_true, y_pred)

def iou(y_true, y_pred):
    intersection = K.sum(y_true * y_pred)
    sum_ = K.sum(y_true + y_pred)
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return jac

def jac_distance(y_true, y_pred):
    y_truef=K.flatten(y_true)
    y_predf=K.flatten(y_pred)
    return - iou(y_true, y_pred)

def precision(y_true,y_pred):
    y_true_pos = K.flatten(y_true)
    y_pred_pos = K.flatten(y_pred)
    true_pos = K.sum(y_true_pos * y_pred_pos)
    #false_neg = K.sum(y_true_pos * (1-y_pred_pos))
    false_pos = K.sum((1-y_true_pos)*y_pred_pos)
    return (true_pos+0.01)/(true_pos+false_pos+0.01)
def recall(y_true,y_pred):
    y_true_pos = K.flatten(y_true)
    y_pred_pos = K.flatten(y_pred)
    true_pos = K.sum(y_true_pos * y_pred_pos)
    false_neg = K.sum(y_true_pos * (1-y_pred_pos))
    #false_pos = K.sum((1-y_true_pos)*y_pred_pos)
    return (true_pos+0.1)/(true_pos+false_neg+0.1)

def f1_score(y_true,y_pred):
  return 2*recall(y_true,y_pred)*precision(y_true,y_pred)/(recall(y_true,y_pred)+precision(y_true,y_pred))

def ACELoss(y_pred, y_true, u=1, a=1, b=1):
    """
    Active Contour Loss
    based on total variations and mean curvature
    """
    def first_derivative(input):
        u = input
        b = u.shape[0]
        m = u.shape[1]
        n = u.shape[2]
        a = u[:, 1, :, :] - u[:, 0, :, :]
        ci_0 = tf.expand_dims(a,1)
        ci_1 = u[:, 2:, :, :] - u[:, 0:254, :, :]
        ci_2 = tf.expand_dims(u[:, -1, :, :] - u[:, 254, :, :],1)
        ci = tf.concat([ci_0, ci_1, ci_2], 1) / 2

        cj_0 = tf.expand_dims(u[:, :, 1, :] - u[:, :, 0, :],2)
        cj_1 = u[:, :, 2:, :] - u[:, :, 0:254, :]
        cj_2 = tf.expand_dims(u[:, :, -1, :] - u[:, :, 254, :],2)
        cj = tf.concat([cj_0, cj_1, cj_2], 2) / 2

        return ci, cj

    def second_derivative(input, ci, cj):
        u = input
        m = u.shape[1]
        n = u.shape[2]

        cii_0 = tf.expand_dims(u[:, 1, :, :] + u[:, 0, :, :] -
                 2 * u[:, 0, :, :],1)
        cii_1 = u[:, 2:, :, :] + u[:, :-2, :, :] - 2 * u[:, 1:-1, :, :]
        cii_2 = tf.expand_dims(u[:, -1, :, :] + u[:, -2, :, :] -
                 2 * u[:, -1, :, :],1)
        cii = tf.concat([cii_0, cii_1, cii_2], 1)

        cjj_0 = tf.expand_dims(u[:, :, 1, :] + u[:, :, 0, :] -
                 2 * u[:, :, 0, :],2)
        cjj_1 = u[:, :, 2:, :] + u[:, :, :-2, :] - 2 * u[:, :, 1:-1, :]
        cjj_2 = tf.expand_dims(u[:, :, -1, :] + u[:, :, -2, :] -
                 2 * u[:, :, -1, :],2)

        cjj = tf.concat([cjj_0, cjj_1, cjj_2], 2)

        cij_0 = ci[:, :, 1:256, :]
        cij_1 = tf.expand_dims(ci[:, :, -1, :],2)

        cij_a = tf.concat([cij_0, cij_1], 2)
        cij_2 = tf.expand_dims(ci[:, :, 0, :],2)
        cij_3 = ci[:, :, 0:255, :]
        cij_b = tf.concat([cij_2, cij_3], 2)
        cij = cij_a - cij_b

        return cii, cjj, cij

    def region(y_pred, y_true, u=1):
        label = y_true#.float()
        
        c_in = tf.ones_like(y_pred)
        c_out = tf.zeros_like(y_pred)
        region_in = K.abs(K.sum(y_pred * ((label - c_in) ** 2)))
        region_out = K.abs(
            K.sum((1 - y_pred) * ((label - c_out) ** 2)))
        region = u * region_in + region_out
        return region

    def elastica(input, a=1, b=1):
        ci, cj = first_derivative(input)
        cii, cjj, cij = second_derivative(input, ci, cj)
        beta = 1e-8
        length = K.sqrt(beta + ci ** 2 + cj ** 2)
        curvature = (beta + ci ** 2) * cjj + \
                    (beta + cj ** 2) * cii - 2 * ci * cj * cij
        curvature = K.abs(curvature) / ((ci ** 2 + cj ** 2) ** 1.5 + beta)
        elastica = K.sum((a + b * (curvature ** 2)) * K.abs(length))
        return elastica

    loss = region(y_pred, y_true, u=u) + elastica(y_pred, a=a, b=b)
    return loss/50000

def new_loss(y_true,y_pred):
  return ACELoss(y_pred,y_true) + dice_loss(y_true,y_pred)