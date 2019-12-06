import tensorflow as tf


# Default hyperparameters:
hparams = tf.contrib.training.HParams(

    # Model:
    # TODO: add more configurable hparams

    # input features: (latency, r, w, r/w, i/o)
    num_features = 5,

    # OP classes + blank + others (10)
    num_classes = 10,

    # Hyper-parameters
    num_hidden = 128,

    # Training:
    #num_epochs = 100,
    num_layers = 1,
    batch_size = 1,
)


def hparams_debug_string():
    values = hparams.values()
    hp = ['  %s: %s' % (name, values[name]) for name in sorted(values)]
    return 'Hyperparameters:\n' + '\n'.join(hp)

