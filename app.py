import gradio as gr
from huggingface_hub import InferenceClient
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
# Load your fine-tuned model and tokenizer
model_name = "kush19/Llama-2-7b-chat-finetune"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Set offload folder
offload_folder = "./offload"  

# Load the model with offload folder specified
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto",  # Automatically assigns layers to devices (CPU/GPU)
    offload_folder=offload_folder,  # This allows weights to be offloaded to disk when necessary
)
"""
For more information on `huggingface_hub` Inference API support, please check the docs: https://huggingface.co/docs/huggingface_hub/v0.22.2/en/guides/inference
"""
# client = InferenceClient("HuggingFaceH4/zephyr-7b-beta")

# Define a function for the chatbot
def respond1(message, history: list[tuple[str, str]], system_message, max_tokens, temperature, top_p):
    inputs = tokenizer.encode(message + tokenizer.eos_token, return_tensors="pt").to(model.device)
    response = model.generate(inputs, max_length=max_tokens, temperature=temperature, top_p=top_p, pad_token_id=tokenizer.eos_token_id)
    response_text = tokenizer.decode(response[:, inputs.shape[-1]:][0], skip_special_tokens=True)
    
    if isinstance(response_text, (list, tuple)):
        response_text = response_text[0] if response_text else ""
    response_text = str(response_text)

    # Append response to the chat history
    history.append((message, response_text))
    return response_text, history

def respond2(message, history, system_message, max_tokens, temperature, top_p):
    # Prepare the input
    full_message = f"{system_message}\n\nHuman: {message}\nAI:"
    inputs = tokenizer.encode(full_message, return_tensors="pt").to(model.device)

    # Generate the response
    response = model.generate(
        inputs, 
        max_length=inputs.shape[1] + max_tokens,
        temperature=temperature, 
        top_p=top_p, 
        pad_token_id=tokenizer.eos_token_id
    )

    # Decode the response
    response_text = tokenizer.decode(response[0][inputs.shape[1]:], skip_special_tokens=True)
    
    # Ensure response_text is a string
    response_text = str(response_text).strip()

    # Return the response (not as a list or tuple)
    return response_text


def respond3(message, history, system_message, max_tokens, temperature, top_p):
    # Prepare the input, including the conversation history
    full_message = f"{system_message}\n\n"

    # Add previous messages to the full_message
    for past_message, past_response in history:
        if past_message:
            full_message += f"Human: {past_message}\n"
        if past_response:
            full_message += f"AI: {past_response}\n"

    # Add the current user message
    full_message += f"Human: {message}\nAI:"

    # Encode the input for the model
    inputs = tokenizer.encode(full_message, return_tensors="pt").to(model.device)

    # Generate the response
    response = model.generate(
        inputs, 
        max_length=inputs.shape[1] + max_tokens,
        temperature=temperature, 
        top_p=top_p, 
        pad_token_id=tokenizer.eos_token_id
    )

    # Decode the response
    response_text = tokenizer.decode(response[0][inputs.shape[1]:], skip_special_tokens=True)
    
    # Ensure response_text is a string
    response_text = str(response_text).strip()

    # Return the response (not as a list or tuple)
    return response_text


# def respond(
#     message,
#     history: list[tuple[str, str]],
#     system_message,
#     max_tokens,
#     temperature,
#     top_p,
# ):
#     messages = [{"role": "system", "content": system_message}]

#     for val in history:
#         if val[0]:
#             messages.append({"role": "user", "content": val[0]})
#         if val[1]:
#             messages.append({"role": "assistant", "content": val[1]})

#     messages.append({"role": "user", "content": message})

#     response = ""

#     for message in client.chat_completion(
#         messages,
#         max_tokens=max_tokens,
#         stream=True,
#         temperature=temperature,
#         top_p=top_p,
#     ):
#         token = message.choices[0].delta.content

#         response += token
#         yield response

"""
For information on how to customize the ChatInterface, peruse the gradio docs: https://www.gradio.app/docs/chatinterface
"""
# demo = gr.ChatInterface(
#     respond,
#     additional_inputs=[
#         gr.Textbox(value="You are a friendly Chatbot.", label="System message"),
#         gr.Slider(minimum=1, maximum=2048, value=512, step=1, label="Max new tokens"),
#         gr.Slider(minimum=0.1, maximum=4.0, value=0.7, step=0.1, label="Temperature"),
#         gr.Slider(
#             minimum=0.1,
#             maximum=1.0,
#             value=0.95,
#             step=0.05,
#             label="Top-p (nucleus sampling)",
#         ),
#     ],
# )

# Create a Gradio app
demo = gr.ChatInterface(
    fn=respond3,
    additional_inputs=[
        gr.Textbox(
            value=(
                "You are an AI trained to engage users using the Socratic method, "
                "guiding them with questions that encourage self-discovery and critical thinking. "
                "Your primary goal is to ask open-ended questions that help the user explore their thoughts and ideas. "
                "However, if the user expresses confusion or reaches a dead end, you should provide clear, concise answers "
                "to help them move forward. Strive to balance inquiry with support, ensuring that the user feels "
                "encouraged and informed throughout the conversation."
            ),
            label="System message"
        ),
        gr.Slider(minimum=1, maximum=2048, value=100, step=1, label="Max new tokens"),
        gr.Slider(minimum=0.1, maximum=4.0, value=0.3, step=0.1, label="Temperature"),
        gr.Slider(minimum=0.1, maximum=1.0, value=0.70, step=0.05, label="Top-p (nucleus sampling)"),
    ],
)


if __name__ == "__main__":
    demo.launch(share=True,server_name="0.0.0.0", server_port=7860)
