import tensorflow as tf
from transformers import TFAutoModelForSequenceClassification, AutoTokenizer

def load_model(model_weights_path: str):
    opt = tf.keras.optimizers.Adam(learning_rate=1e-5, weight_decay=0.01)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    model = TFAutoModelForSequenceClassification.from_pretrained("prajjwal1/bert-mini", num_labels=6, from_pt=True)
    model.compile(optimizer=opt, loss=loss_fn, metrics=['accuracy'])

    model.load_weights(model_weights_path)

    return model


def predict_sentiment(review_text: str):
    model = load_model()

    tokenizer = AutoTokenizer.from_pretrained("prajjwal1/bert-mini")
    encodings = tokenizer(list(review_text), truncation=True, padding=True, max_length=200)

    ds = tf.data.Dataset.from_tensor_slices((
        dict(encodings),
        list([0])
    ))

    y_pred = model.predict(ds.batch(1)) 
    y_pred = tf.nn.softmax(y_pred.logits) 
    y_pred = tf.argmax(y_pred, axis=1) 

    return y_pred
