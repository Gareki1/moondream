# import os
# os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

import torch
import certifi
import requests
import argparse
from PIL import Image
from moondream import Moondream, detect_device, LATEST_REVISION
from queue import Queue
from threading import Thread
from transformers import TextIteratorStreamer, AutoTokenizer


api_key = '4d28d85a7cd44229b7fa7f64dddf274c'
endpoint = 'https://api.cognitive.microsofttranslator.com/translate?api-version=3.0'

def translate(text, lang):
    if not text or not lang:
        raise ValueError("Text and target language are required")

    try:
        response = requests.post(
            f"{endpoint}&to={lang}",
            json=[{'Text': text}],
            headers={
                'Ocp-Apim-Subscription-Key': api_key,
                'Ocp-Apim-Subscription-Region': 'southeastasia',
                'Content-Type': 'application/json'
            },
            verify=certifi.where()
        )
        

        response.raise_for_status()  # Raise an error for bad responses
        translated_text = response.json()[0]['translations'][0]['text']
        return translated_text

    except requests.exceptions.RequestException as error:
        print('Error translating text:', error)
        return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--prompt", type=str, required=False)
    parser.add_argument("--caption", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    if args.cpu:
        device = torch.device("cpu")
        dtype = torch.float32
    else:
        device, dtype = detect_device()
        if device != torch.device("cpu"):
            print("Using device:", device)
            print("If you run into issues, pass the `--cpu` flag to this script.")
            print()

    image_path = args.image
    prompt = args.prompt

    model_id = "vikhyatk/moondream2"
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=LATEST_REVISION)
    moondream = Moondream.from_pretrained(
        model_id,
        revision=LATEST_REVISION,
        torch_dtype=dtype,
    ).to(device=device)
    moondream.eval()

    image = Image.open(image_path)

    if args.caption:
        print(moondream.caption(images=[image], tokenizer=tokenizer)[0])
    else:
        image_embeds = moondream.encode_image(image)
        lang = input("What language do you want ? EX.Hindi(hi),Marathi(mr),Spanish(es)")
        if prompt is None:
            chat_history = ""

            while True:
                question = input("> ")

                result_queue = Queue()

                streamer = TextIteratorStreamer(tokenizer, skip_special_tokens=True)
                # Separate direct arguments from keyword arguments
                thread_args = (image_embeds, question, tokenizer, chat_history)
                thread_kwargs = {"streamer": streamer, "result_queue": result_queue}

                thread = Thread(
                    target=moondream.answer_question,
                    args=thread_args,
                    kwargs=thread_kwargs,
                )
                thread.start()
                sentence = str()
                buffer = ""
                for new_text in streamer:
                    buffer += new_text
                    if not new_text.endswith("<") and not new_text.endswith("END"):
                        sentence += new_text   
                        print(buffer, end="", flush=True)
                        buffer = ""
               # print(sentence)
                if lang=='en':
                    print(sentence)
                else:
                    print("Translating...\n")
                    trans_answer = translate(sentence, lang)
                    print(trans_answer)

                thread.join()

                answer = result_queue.get()
                chat_history += f"Question: {question}\n\nAnswer: {answer}\n\n"
        else:
            print(">", prompt)
            answer = moondream.answer_question(image_embeds, prompt, tokenizer)
            print(answer)
            if lang=='en':
                print(answer)
            else:
                print("Translating...\n")
                trans_answer = translate(answer, lang)
                print(trans_answer)
