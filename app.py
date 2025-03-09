import os
import cv2
import numpy as np
import tensorflow as tf
from flask import Flask, request, render_template
import matplotlib.pyplot as plt
import io
import base64

app = Flask(__name__)

# Load the trained model
model = tf.keras.models.load_model("best_model.keras")

# Extract layers for feature maps (excluding dense layers)
layer_outputs = [layer.output for layer in model.layers if "conv" in layer.name or "pool" in layer.name]
feature_model = tf.keras.models.Model(inputs=model.inputs, outputs=layer_outputs)

UPLOAD_FOLDER = "static"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def generate_feature_maps(feature_maps):
    """Generate base64-encoded images of feature maps."""
    feature_images = []
    for i, fmap in enumerate(feature_maps):
        fmap = np.squeeze(fmap, axis=0)  # Remove batch dimension
        num_filters = fmap.shape[-1]
        
        fig, axes = plt.subplots(1, min(num_filters, 6), figsize=(15, 5))  # Show up to 6 filters
        for j in range(min(num_filters, 6)):
            axes[j].imshow(fmap[:, :, j], cmap="viridis")  # Visualize activation map
            axes[j].axis("off")
        
        # Convert the plot to a base64-encoded image
        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        plt.close(fig)
        buf.seek(0)
        feature_images.append(base64.b64encode(buf.read()).decode("utf-8"))
    
    return feature_images

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    feature_images = []
    
    if request.method == "POST":
        file = request.files["file"]
        if file:
            uploaded_path = os.path.join(UPLOAD_FOLDER, "uploaded.png")
            file.save(uploaded_path)

            # Preprocess the image
            img = cv2.imread(uploaded_path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (28, 28))  
            img = cv2.bitwise_not(img)
            img = img.astype("float32") / 255.0  
            img = np.expand_dims(img, axis=[0, -1])  

            # Get feature maps
            feature_maps = feature_model.predict(img)
            feature_images = generate_feature_maps(feature_maps)

            # Get prediction
            pred = model.predict(img)
            prediction = np.argmax(pred)

            return render_template("index.html", 
                                   uploaded_image="uploaded.png",
                                   feature_images=feature_images,
                                   prediction=prediction)

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)