from keras.layers import Input, Embedding, Dense, Flatten, Dropout, \
    concatenate, BatchNormalization, SpatialDropout1D, Activation
from keras.layers.advanced_activations import LeakyReLU, PReLU, ELU
from keras.callbacks import Callback, LearningRateScheduler, EarlyStopping
from keras.initializers import RandomUniform
from keras.regularizers import l1, l2, l1_l2
from keras.models import Model
from keras.optimizers import Adam
import numpy as np
from sklearn.metrics import roc_auc_score
import logging
module_logger = logging.getLogger(__name__)


def get_model(nn_params, X_train, X_val, X_test, categorical_features):
    train_dict = {}
    valid_dict = {}
    test_dict = {}
    feature_input_list = []
    embed_nodes_list = []
    numerical_features = []
    total_cate_embedding_dimension = 0
    for col in X_train:
        if col not in categorical_features:
            numerical_features.append(col)

    if len(categorical_features) > 0:
        for col in categorical_features:
            train_dict[col] = np.array(X_train[col])
            valid_dict[col] = np.array(X_val[col])
            test_dict[col] = np.array(X_test[col])
            cat_input = Input(shape=(1,), name=col)
            feature_input_list.append(cat_input)
            embed_input_dim = np.max([X_train[col].max(), X_val[col].max(), X_test[col].max()]) + 1
            embed_output_dimension = nn_params[
                'cate_embedding_dimension']  # np.min([cate_embedding_dimension, embed_input_dim])
            total_cate_embedding_dimension += embed_output_dimension
            module_logger.debug('prepare col [{}] for NN'.format(col))
            module_logger.debug('embed input dimension: {}'.format(embed_input_dim))
            module_logger.debug('embed output dimension: {}'.format(embed_output_dimension))
            embed_node = Embedding(embed_input_dim, embed_output_dimension)(cat_input)
            embed_nodes_list.append(embed_node)
        embed_layer = concatenate(embed_nodes_list)
        dropout_layer = SpatialDropout1D(nn_params['drop_rate'])(embed_layer)
        categorical_node = Flatten()(dropout_layer)

    if len(numerical_features) > 0:
        key_for_numerical_features = 'numerical_features'
        train_dict[key_for_numerical_features] = X_train[numerical_features].values
        valid_dict[key_for_numerical_features] = X_val[numerical_features].values
        test_dict[key_for_numerical_features] = X_test[numerical_features].values
        num_input = Input(shape=(len(numerical_features),), name=key_for_numerical_features)
        feature_input_list.append(num_input)
        numerical_node = num_input
        for i in range(nn_params['dense_numerical_n_layers']):
            numerical_node = functional_dense(nn_params['dense_units'], numerical_node,
                                              batch_norm=True, act='relu',
                                              dropout=nn_params['drop_rate'], name='numerical_{}'.format(i))

    if len(numerical_features) > 0 and len(categorical_features) > 0:
        x = concatenate([categorical_node, numerical_node])
    elif len(numerical_features) > 0:
        x = numerical_node
    elif len(categorical_features) > 0:
        x = categorical_node
    else:
        return 0  # raise exception?

    for i in range(nn_params['combined_dense_n_layers']):
        x = functional_dense(nn_params['dense_units'], x, dropout=nn_params['drop_rate'], name='combined_{}'.format(i))

    nn_final_outputs = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=feature_input_list, outputs=nn_final_outputs)

    def exp_decay(init, fin, steps):
        return (init / fin) ** (1 / (steps - 1)) - 1

    steps_in_one_epoch = int(len(X_train) / nn_params['batch_size'])
    steps_for_lr_decay = steps_in_one_epoch * nn_params['epochs_for_lr']
    lr_decay = exp_decay(nn_params['lr_init'], nn_params['lr_fin'], steps_for_lr_decay)
    module_logger.debug('steps in on epoch: {}'.format(steps_in_one_epoch))
    module_logger.debug('steps for lr decay: {}'.format(steps_for_lr_decay))
    module_logger.debug('lr decay: {:.6f}'.format(lr_decay))
    optimizer = Adam(lr=nn_params['lr_init'], decay=lr_decay)

    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    model.summary()

    return model, train_dict, valid_dict, test_dict


def functional_dense(dense_units, x, batch_norm=False, act='relu', lw1=0.0, dropout=0, name=''):
    if lw1 == 0.0:
        x = Dense(dense_units, name=name + '_dense')(x)
    else:
        x = Dense(dense_units, kernel_regularizer=l1(lw1), name=name + '_dense')(x)

    if batch_norm:
        x = BatchNormalization(name=name + '_batchnorm')(x)

    if act in {'relu', 'tanh', 'sigmoid'}:
        x = Activation(act, name=name + '_activation')(x)
    elif act == 'prelu':
        x = PReLU(name=name + '_activation')(x)
    elif act == 'leakyrelu':
        x = LeakyReLU(name=name + '_activation')(x)
    elif act == 'elu':
        x = ELU(name=name + '_activation')(x)

    if dropout > 0:
        x = Dropout(dropout, name=name + '_dropout')(x)

    return x


class LearningRateTracker(Callback):
    def __init__(self, include_on_batch=False):
        self.logger = logging.getLogger(__name__ + '.' + self.__class__.__name__)
        super(LearningRateTracker, self).__init__()
        self.include_on_batch = include_on_batch

    def on_batch_end(self, batch, logs={}):
        if self.include_on_batch:
            self._show_lr('batch')

    def on_epoch_end(self, epoch, logs={}):
        self._show_lr('epoch')

    def _show_lr(self, log_flag):
        from keras import backend as K
        lr_init = self.model.optimizer.lr
        lr_decay = self.model.optimizer.decay
        iterations = self.model.optimizer.iterations
        lr_current = lr_init / (1. + lr_decay * K.cast(iterations, K.dtype(lr_decay)))
        self.logger.debug('After this {}: init lr {:.6f}, lr_decay {:.6f}, '
                          'iterations {}, current lr {:.6f}'
                          .format(log_flag, K.eval(lr_init), K.eval(lr_decay),
                                  K.eval(iterations), K.eval(lr_current)))
        # Use the following if you are using LearningRateScheduler in callbacks
        # print('\nlearning rate ', K.eval(self.model.optimizer.lr), '\n')


class RocAucMetricCallback(Callback):
    def __init__(self, validation_data=(), predict_batch_size=100000, include_on_batch=False):
        super(RocAucMetricCallback, self).__init__()
        self.predict_batch_size = predict_batch_size
        self.include_on_batch = include_on_batch
        self.X_val, self.y_val = validation_data

    def on_batch_begin(self, batch, logs={}):
        pass

    def on_batch_end(self, batch, logs={}):
        if self.include_on_batch:
            self._compute_auc(logs)

    def on_train_begin(self, logs={}):
        if not ('roc_auc_val' in self.params['metrics']):
            self.params['metrics'].append('roc_auc_val')

    def on_train_end(self, logs={}):
        pass

    def on_epoch_begin(self, epoch, logs={}):
        pass

    def on_epoch_end(self, epoch, logs={}):
        self._compute_auc(logs)

    def _compute_auc(self, logs):
        logs['roc_auc_val'] = float('-inf')
        if self.validation_data:
            logs['roc_auc_val'] = \
                roc_auc_score(self.y_val,
                              self.model.predict(self.X_val,
                                                 batch_size=self.predict_batch_size))
        return logs
