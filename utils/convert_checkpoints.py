import tensorflow as tf
import tensorflow.compat.v1 as tf1

def print_checkpoint(save_path):
  reader = tf.train.load_checkpoint(save_path)
  shapes = reader.get_variable_to_shape_map()
  dtypes = reader.get_variable_to_dtype_map()
  print(f"Checkpoint at '{save_path}':")
  for key in shapes:
    print(f"  (key='{key}', shape={shapes[key]}, dtype={dtypes[key].name}, "
          f"value={reader.get_tensor(key)})")

def convert_tf2_to_tf1(checkpoint_path, output_prefix):
    """Converts a TF2 checkpoint to TF1.
    """
    vars = {}
    reader = tf.train.load_checkpoint(checkpoint_path)
    dtypes = reader.get_variable_to_dtype_map()
    for key in dtypes.keys():
      # Get the "name" from the 
      if key.startswith('var_list/'):
        var_name = key.split('/')[1]
        # TF2 checkpoint keys use '/', so if they appear in the user-defined name,
        # they are escaped to '.S'.
        var_name = var_name.replace('.S', '/')
        vars[var_name] = tf.Variable(reader.get_tensor(key))

    return tf1.train.Saver(var_list=vars).save(sess=None, save_path=output_prefix)

# Make sure to run the snippet in `Save a TF2 checkpoint in TF1`.
tf2ckpt = '../gcp_checkpoints/WQ_muller_tpu.yaml.ckpt'
tf1ckpt = '../gcp_checkpoints/WQ_muller_tpu_tf1'
print_checkpoint(tf2ckpt)
converted_path = convert_tf2_to_tf1(tf2ckpt, tf1ckpt)
print("\n[Converted]")
print_checkpoint(converted_path)

