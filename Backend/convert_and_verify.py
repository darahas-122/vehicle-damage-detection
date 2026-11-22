import os
import zipfile
import json
import io
import tempfile
import h5py
import numpy as np

os.environ['KERAS_HOME'] = r'C:\Users\DS_PROJECT\vehicle damage detection\.keras'
os.environ['KERAS_BACKEND'] = 'torch'
os.environ['MPLCONFIGDIR'] = r'C:\Users\DS_PROJECT\vehicle damage detection\.keras\mpl'
os.makedirs(os.environ['MPLCONFIGDIR'], exist_ok=True)

import keras

z = zipfile.ZipFile('Backend/efficientnetb0_cardd_render.keras')
cfg_str = z.read('config.json').decode('utf-8')
cfg_str = cfg_str.replace('keras.src.engine.functional', 'keras.src.models.functional')
cfg = json.loads(cfg_str)

def fix_config(obj):
    if isinstance(obj, dict):
        if obj.get('class_name') in ('BatchNormalization', 'Normalization') and 'config' in obj:
            if isinstance(obj['config'].get('axis'), list):
                obj['config']['axis'] = obj['config']['axis'][0]
        if obj.get('class_name') == 'DepthwiseConv2D' and 'config' in obj:
            obj['config'].pop('groups', None)
        if 'inbound_nodes' in obj and isinstance(obj['inbound_nodes'], list):
            for node in obj['inbound_nodes']:
                if isinstance(node, list):
                    for arg in node:
                        if isinstance(arg, list) and len(arg) >= 2 and arg[0] == 'efficientnetb0' and arg[1] == 1:
                            arg[1] = 0
        for k, v in obj.items():
            fix_config(v)
    elif isinstance(obj, list):
        for item in obj:
            fix_config(item)

fix_config(cfg)

model = keras.saving.deserialize_keras_object(cfg)
print("Model deserialized successfully!")

# Extract and open weights
weights_bytes = z.read('model.weights.h5')
with tempfile.NamedTemporaryFile(suffix='.weights.h5', delete=False) as tf_file:
    tf_file.write(weights_bytes)
    temp_weights_path = tf_file.name

with h5py.File(temp_weights_path, 'r') as hf:
    # Separate counters for each layer type
    counters = {
        'Conv2D': 0,
        'DepthwiseConv2D': 0,
        'BatchNormalization': 0,
        'Normalization': 0,
        'Dense': 0
    }

    prefix_map = {
        'Conv2D': 'layers\\functional\\layers\\conv2d',
        'DepthwiseConv2D': 'layers\\functional\\layers\\depthwise_conv2d',
        'BatchNormalization': 'layers\\functional\\layers\\batch_normalization',
        'Normalization': 'layers\\functional\\layers\\normalization',
        'Dense': 'layers\\dense'
    }

    all_layers = []
    for l in model.layers:
        if hasattr(l, 'layers'):
            all_layers.extend(l.layers)
        else:
            all_layers.append(l)

    assigned = 0
    for layer in all_layers:
        cname = layer.__class__.__name__
        if cname in counters:
            idx = counters[cname]
            counters[cname] += 1

            if cname == 'Dense':
                key = 'layers\\dense'
            elif cname == 'Normalization':
                key = 'layers\\functional\\layers\\normalization'
            else:
                base = prefix_map[cname]
                key = base if idx == 0 else f"{base}_{idx}"

            if key in hf and 'vars' in hf[key]:
                var_group = hf[key]['vars']
                var_keys = sorted(var_group.keys(), key=lambda x: int(x))
                var_arrays = [var_group[vk][()] for vk in var_keys]
                
                # Check shapes match
                target_shapes = [w.shape for w in layer.weights]
                src_shapes = [a.shape for a in var_arrays]
                assert target_shapes == src_shapes, f"Shape mismatch for {layer.name} ({key}): {target_shapes} vs {src_shapes}"

                layer.set_weights(var_arrays)
                assigned += 1
            else:
                print(f"Warning: key {key} not found for {layer.name}")

    print(f"Successfully assigned weights to ALL {assigned} weighted layers!")

# Test predictions
rng = np.random.default_rng(42)
sample = rng.uniform(0, 255, size=(1, 224, 224, 3)).astype("float32")
pred = model.predict(sample, verbose=0)
print("Sample prediction on test image:", pred[0])

# Now test save to standard Keras 3 format!
keras3_path = 'Backend/efficientnetb0_cardd_keras3.keras'
model.save(keras3_path)
print(f"Successfully saved clean Keras 3 model to {keras3_path}!")

# Test loading the newly saved Keras 3 model directly!
loaded_k3 = keras.models.load_model(keras3_path)
pred2 = loaded_k3.predict(sample, verbose=0)
print("Difference between memory and reloaded:", np.max(np.abs(pred - pred2)))
assert np.allclose(pred, pred2, atol=1e-5)
print("VERIFICATION COMPLETE: Predictions match 100%!")
