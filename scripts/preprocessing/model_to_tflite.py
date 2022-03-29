import tensorflow as tf

# Convert the model
# tf.enable_control_flow_v2()
converter = tf.lite.TFLiteConverter.from_saved_model('/home/whoisltd/detect/training/exported-models/my_model/saved_model') # path to the SavedModel directory
converter.target_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
converter.allow_custom_ops=True
converter.experimental_new_converter =True
tflite_model = converter.convert()

# Save the model.
with open('model.tflite', 'wb') as f:
  f.write(tflite_model)