"""Convert Keras model to tensorflow format for serving
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1" 

import tensorflow as tf

import sgidspace.sgikeras.metrics as sgimetrics

# The export path contains the name and the version of the model
tf.keras.backend.set_learning_phase(0)  # Ignore dropout at inference

deps = {
     'precision': sgimetrics.precision,
     'recall': sgimetrics.recall,
     'fmeasure': sgimetrics.fmeasure,
}
model = tf.keras.models.load_model('epoch3.hdf5', custom_objects=deps)
export_path = 'serving'

# Fetch the Keras session and save the model
# The signature definition is defined by the input and output tensors
# And stored with the default serving key
with tf.keras.backend.get_session() as sess:
    tf.saved_model.simple_save(
        sess,
        export_path,
        inputs={'input_seq_batch': model.input},
        outputs={t.name: t for t in model.outputs})


