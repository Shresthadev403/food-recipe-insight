from flask import Flask, request, jsonify
from transformers import GPT2LMHeadModel, GPT2Tokenizer, AutoImageProcessor, AutoModelForImageClassification
import torch

app = Flask(__name__)

# Functions
def convert_tokens_to_string(tokens, tokenizer):
    """
    Converts a sequence of tokens (string) into a single string.
    """
    if tokens is None:
        return ""

    cleaned_tokens = [token for token in tokens if token is not None]
    text = tokenizer.decode(cleaned_tokens, skip_special_tokens=True) if cleaned_tokens else ""

    return text

def generate_text(tokenizer_dir, model_dir, prompt):
    # Load the fine-tuned model
    model = GPT2LMHeadModel.from_pretrained(model_dir)
    tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_dir)
    # Get the custom EOS token ID from the tokenizer
    custom_eos_token_id = tokenizer.encode('<RECIPE_END>', add_special_tokens=False)[0]
    # Set the custom EOS token ID in the model configuration
    model.config.eos_token_id = custom_eos_token_id

    model.eval()

    # Generate text
    input_ids = tokenizer.encode(prompt, return_tensors='pt')
    attention_mask = torch.ones_like(input_ids)
    output = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_length=500, num_return_sequences=1)

    # Convert tokens to string
    generated_text = convert_tokens_to_string(output[0], tokenizer)

    # Replace '<' with '\n<' to create a new line at each tag
    generated_text = generated_text.replace('<', '\n<')

    return generated_text

# Load models outside of the route functions
image_processor = AutoImageProcessor.from_pretrained("./food-image-classification")
image_model = AutoModelForImageClassification.from_pretrained("./food-image-classification")
text_generator_model = GPT2LMHeadModel.from_pretrained("./controlled-food-recipe-generation")
text_generator_tokenizer = GPT2Tokenizer.from_pretrained("./controlled-food-recipe-generation")

# Routes
@app.route('/', methods=['GET'])
def get_info():
    return jsonify({'message': "Flask app running successfully"})

@app.route('/generate_recipe', methods=['POST'])
def generate_recipe():
    data = request.json
    prompt = data['prompt']

    # Generate text using the fine-tuned model
    generated_text = generate_text("/path/to/controlled-food-recipe-generation", "/path/to/controlled-food-recipe-generation", prompt)

    return jsonify({'generated_text': generated_text})

@app.route('/classify', methods=['POST'])
def classify_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'})

    image_file = request.files['image']
    inputs = image_processor(image_file, return_tensors="pt")

    with torch.no_grad():
        logits = image_model(**inputs).logits
    predicted_label_id = logits.argmax(-1).item()
    predicted_label = image_model.config.id2label[predicted_label_id]

    return jsonify({'predicted_label': predicted_label})

if __name__ == "__main__":
    app.run(debug=True)
