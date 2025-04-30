import os

import tensorflow as tf
import kagglehub as kh
import pandas as pd
import numpy as np

from transformers import TFAutoModelForSequenceClassification, AutoTokenizer, create_optimizer
from sklearn.model_selection import GroupShuffleSplit, train_test_split

tf.config.list_physical_devices('GPU')

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True) # Set GPU memory growth, if exists
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(len(gpus), 'Physical GPUs,', len(logical_gpus), 'Logical GPUs')
    except RuntimeError as e:
        print(e)

tf.get_logger().setLevel('ERROR')
tf.random.set_seed(42)
tf.keras.mixed_precision.set_global_policy('mixed_float16')

path = kh.dataset_download("mohamedbakhet/amazon-books-reviews")

df_ratings = pd.read_csv(os.path.join(path, 'Books_rating.csv'))
df_data = pd.read_csv(os.path.join(path, 'books_data.csv'))

df_sentiment = df_ratings.copy()
df_sentiment = df_sentiment.drop(['Title', 'Price', 'User_id', 'profileName', 'review/helpfulness', 'review/time', 'review/summary'], axis=1)

df_sentiment['review/text'] = df_sentiment['review/text'].str.lower().str.strip().str.replace(r'[\n\t]', ' ')

df_sentiment = df_sentiment.drop_duplicates(subset='review/text')

n_samples_per_class = 40_000
df = df_sentiment.groupby('review/score', group_keys=False).apply(
    lambda x: x.sample(n=n_samples_per_class, random_state=42)
).reset_index(drop=True)

group_kfold = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

train_set, test_set = None, None
for train_index, test_index in group_kfold.split(df, df['review/score'], groups=df['Id']):
    train_set = df.iloc[train_index]
    test_set = df.iloc[test_index]

X_train = train_set['review/text'].tolist()
y_train = train_set['review/score'].tolist()
X_test = test_set['review/text'].tolist()
y_test = test_set['review/score'].tolist()

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

tokenizer = AutoTokenizer.from_pretrained("prajjwal1/bert-mini")

train_encodings = tokenizer(X_train, truncation=True, padding=True, max_length=200)
val_encodings = tokenizer(X_val, truncation=True, padding=True, max_length=200)
test_encodings = tokenizer(X_test, truncation=True, padding=True, max_length=200)

train_dataset = tf.data.Dataset.from_tensor_slices((
    dict(train_encodings),
    y_train
))
val_dataset = tf.data.Dataset.from_tensor_slices((
    dict(val_encodings),
    y_val
))
test_dataset = tf.data.Dataset.from_tensor_slices((
    dict(test_encodings),
    y_test
))

batch_size = 64
num_epochs = 20

def build_model(weight_decay, learning_rate=1e-5, num_labels=6):
    model = TFAutoModelForSequenceClassification.from_pretrained("prajjwal1/bert-mini", num_labels=6, from_pt=True)

    steps_per_epoch = len(train_dataset) // batch_size
    num_train_steps = steps_per_epoch*num_epochs
    num_warmup_steps = 0

    optimizer, _ = create_optimizer(
        init_lr=learning_rate,
        num_train_steps=num_train_steps,
        num_warmup_steps=num_warmup_steps,
        weight_decay_rate=weight_decay
    )

    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )
    
    return model

weight_decay_values = sorted([
    0.0, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1
])
results = {}

for wd in weight_decay_values:
    print(f'Training with weight decay = {wd}')
    model = build_model(weight_decay=wd)

    history=model.fit(
        train_dataset.shuffle(1000).batch(batch_size),
        validation_data=val_dataset.shuffle(1000).batch(batch_size),
        epochs=num_epochs,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)
        ],
        verbose=2
    )

    val_acc = max(history.history['val_accuracy'])
    results[wd] = val_acc
    print(f'Weight decay: {wd} -> Best val accuracy: {val_acc:.4f}')

best_wd = max(results, key=results.get)
print(f'Best weight decay: {best_wd} with accuracy {results[best_wd]:.4f}')
