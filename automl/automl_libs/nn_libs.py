from keras.layers import Input, Embedding, Dense, Flatten, Dropout, \
    concatenate, BatchNormalization, SpatialDropout1D, Activation
from keras.layers.advanced_activations import LeakyReLU, PReLU, ELU
from keras.callbacks import Callback, LearningRateScheduler, EarlyStopping
from keras.initializers import RandomUniform
from keras.regularizers import l1, l2, l1_l2
from keras.models import Model
from keras.optimizers import Adam
import numpy as np
import pandas as pd
import re
from sklearn.metrics import roc_auc_score, log_loss
import pdb
import logging
module_logger = logging.getLogger(__name__)


def get_model(nn_params, X_train, X_val, X_test, categorical_features):
    """
    Params:
        nn_params: dict
        X_train, X_val, X_test: pandas dataframe (or 2D np array, which will be converted to df)
        categorical_features: list of columns names
    """
    # convert to df just in case they are not df
    X_train = pd.DataFrame(X_train)
    X_test = pd.DataFrame(X_test)
    if X_val is None:
        X_val = X_train.tail(1)
    else:
        X_val = pd.DataFrame(X_val)

    train_dict = {}
    valid_dict = {}
    test_dict = {}
    feature_input_list = []
    embed_nodes_list = []
    numerical_features = []
    # total_cate_embedding_dimension = 0
    for col in X_train:
        if col not in categorical_features:
            numerical_features.append(col)

    # assume all cat columns are label encoded
    if len(categorical_features) > 0:
        embed_outdim = nn_params.get('cat_emb_outdim')  # could be a constant or a dict (col name:embed out dim)
        for col in categorical_features:
            # if col not in ['adid', 'model', 'city', 'creative_type', 'inner_slot_id', 'user_tags_sorted', 'creative_id']:
            #     continue
            # construct data for training, validation and prediction
            train_dict[str(col)] = np.array(X_train[col])  # in case col is not a string, but say, a number
            valid_dict[str(col)] = np.array(X_val[col])
            test_dict[str(col)] = np.array(X_test[col])

            # construct categorical input nodes
            cat_input = Input(shape=(1,), name=str(col))
            feature_input_list.append(cat_input)
            # embed_input_dim = np.max([X_train[col].nunique(), X_val[col].nunique(), X_test[col].nunique()]) + 1
            embed_input_dim = np.max([X_train[col].max(), X_val[col].max(), X_test[col].max()]) + 1
            # why +1 in embed_input_dim: because categorical cols are assumed labelencoded, which start from 0
            # so e.g. if X_train[col].max() returns 3, it means there are 3 + 1 categories: 0,1,2,3

            # parse out the embed out dimension for this column
            if isinstance(embed_outdim, dict):
                embed_output_dimension = embed_outdim[col]
            else:
                embed_output_dimension = np.min([embed_outdim, int(embed_input_dim/2)])
            embed_output_dimension = int(np.log2(embed_input_dim)/np.log2(1.5))
            if embed_output_dimension < 2:
                embed_output_dimension = 2
            # total_cate_embedding_dimension += embed_output_dimension
            module_logger.debug('Col [{}]: embed dim: input {}, output {}'
                                .format(col, embed_input_dim, embed_output_dimension))
            embed_node = Embedding(embed_input_dim,
                                   embed_output_dimension,
                                   # input_length=1,  # depends on the dimension of train_dict[str(col)]
                                                      # if (N,1) then 1, if (N,) then None(default)
                                   trainable=True,
                                   embeddings_regularizer=l2(0.001)
                                   # embeddings_initializer=RandomUniform(minval=-0.005, maxval=0.005),
                                   )(cat_input)
            if nn_params['cat_emb_drop_rate'] > 0:
                embed_node = SpatialDropout1D(nn_params['cat_emb_drop_rate'])(embed_node)
            embed_node = Flatten()(embed_node)
            embed_nodes_list.append(embed_node)
        categorical_node = concatenate(embed_nodes_list)

            # embed_nodes_list.append(embed_node)
        # embed_layer = concatenate(embed_nodes_list)
        # if nn_params['cat_emb_drop_rate'] > 0:
        #     embed_layer = SpatialDropout1D(nn_params['cat_emb_drop_rate'])(embed_layer)
        # categorical_node = Flatten()(embed_layer)

    # numerical_features = []  ##################################################################################
    if len(numerical_features) > 0:
        key_for_numerical_features = 'numerical_features'
        train_dict[key_for_numerical_features] = X_train[numerical_features].values
        valid_dict[key_for_numerical_features] = X_val[numerical_features].values
        test_dict[key_for_numerical_features] = X_test[numerical_features].values
        num_input = Input(shape=(len(numerical_features),), name=key_for_numerical_features)
        feature_input_list.append(num_input)
        numerical_node = num_input
        for i, dense_units in enumerate(nn_params['num_layers_dense_units']):
            numerical_node = functional_dense(dense_units, numerical_node,
                                              batch_norm=True, act='relu',
                                              dropout=nn_params['num_layers_drop_rate'],
                                              name='numerical_{}'.format(i))

    if len(numerical_features) > 0 and len(categorical_features) > 0:
        x = concatenate([categorical_node, numerical_node])
    elif len(numerical_features) > 0:
        x = numerical_node
    elif len(categorical_features) > 0:
        x = categorical_node
    else:
        return 0  # raise exception?

    for i, dense_units in enumerate(nn_params['combined_layers_dense_units']):
        x = functional_dense(dense_units, x, dropout=nn_params['combined_layers_drop_rate'],
                             name='combined_{}'.format(i))

    nn_final_outputs = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=feature_input_list, outputs=nn_final_outputs)

    lr_init = nn_params['lr_init']
    lr_fin = nn_params['lr_fin']
    if lr_init != lr_fin:
        # compute lr decay
        steps_in_one_epoch = int(len(X_train) / nn_params['batch_size'])
        steps_for_lr_decay = steps_in_one_epoch * nn_params['ep_for_lr']

        def exp_decay(init, fin, steps):
            return (init / fin) ** (1 / (steps - 1)) - 1

        lr_decay = exp_decay(lr_init, lr_fin, steps_for_lr_decay)
        module_logger.debug('steps in on epoch: {}'.format(steps_in_one_epoch))
        module_logger.debug('steps for lr decay: {}'.format(steps_for_lr_decay))
        module_logger.debug('lr decay: {:.6f}'.format(lr_decay))
        optimizer = Adam(lr=lr_init, decay=lr_decay)
    else:
        optimizer = Adam(lr=lr_init)

    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    # model.summary()

    return model, train_dict, valid_dict, test_dict


def functional_dense(dense_units, x, batch_norm=False, act='relu', lw1=0.0, dropout=0, name=''):
    if isinstance(dense_units, str):
        module_logger.warning('dense_unit {} is a string (it might be read from csv),'
                              ' converting it to int...'.format(dense_units))
        dense_units = int(dense_units)
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
        self.logger.debug('At {} end: init lr {:.5f}, lr_decay {:.5f}, '
                          'iterations {}, current lr {:.5f}'
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
        if not ('val_auc' in self.params['metrics']):
            self.params['metrics'].append('val_auc')

    def on_train_end(self, logs={}):
        pass

    def on_epoch_begin(self, epoch, logs={}):
        pass

    def on_epoch_end(self, epoch, logs={}):
        self._compute_auc(logs)

    def _compute_auc(self, logs):
        logs['val_auc'] = float('-inf')
        if self.validation_data:
            logs['val_auc'] = \
                roc_auc_score(self.y_val,
                              self.model.predict(self.X_val,
                                                 batch_size=self.predict_batch_size))
        return logs


class LogLossMetricCallback(Callback):
    def __init__(self, validation_data=(), predict_batch_size=100000, include_on_batch=False):
        super(LogLossMetricCallback, self).__init__()
        self.predict_batch_size = predict_batch_size
        self.include_on_batch = include_on_batch
        self.X_val, self.y_val = validation_data

    def on_batch_begin(self, batch, logs={}):
        pass

    def on_batch_end(self, batch, logs={}):
        if self.include_on_batch:
            self._compute_logloss(logs)

    def on_train_begin(self, logs={}):
        if not ('val_logloss' in self.params['metrics']):
            self.params['metrics'].append('val_logloss')

    def on_train_end(self, logs={}):
        pass

    def on_epoch_begin(self, epoch, logs={}):
        pass

    def on_epoch_end(self, epoch, logs={}):
        self._compute_logloss(logs)

    def _compute_logloss(self, logs):
        logs['val_logloss'] = float('inf')
        if self.validation_data:
            logs['val_logloss'] = log_loss(self.y_val, self.model.predict(self.X_val,
                                                                          batch_size=self.predict_batch_size))
        return logs

