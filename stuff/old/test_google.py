"""
handles google PaLM api calls
also known as a waste of a supercomputer
"""

import google.generativeai as palm
from config import api_keys


def ask_palm(prompt, messages):
    palm.configure(api_key=api_keys.google)

    defaults = {
        'model': 'models/text-bison-001',
        'temperature': 0.5,
        'candidate_count': 3,
        'top_k': 40,
        'top_p': 0.95,
        'max_output_tokens': 4096,
        'stop_sequences': [],
        'safety_settings': [{"category": "HARM_CATEGORY_DEROGATORY", "threshold": 3},
                            {"category": "HARM_CATEGORY_TOXICITY", "threshold": 3},
                            {"category": "HARM_CATEGORY_VIOLENCE", "threshold": 3},
                            {"category": "HARM_CATEGORY_SEXUAL", "threshold": 3},
                            {"category": "HARM_CATEGORY_MEDICAL", "threshold": 3},
                            {"category": "HARM_CATEGORY_DANGEROUS", "threshold": 3}],
    }
    response = palm.generate_text(
        **defaults,
        prompt=messages)
    if response.result is None:
        print('good job you fucked it up, no response')
        print('safety info:')
        print(response.safety_feedback)
        print('filter info:')
        print(response.filters)

    print(response.result)
    return response


def chat_palm(prompt, messages, examples):
    # TODO: dynamic control of examples from chat
    # TODO: messages handling/formatting chat history, handle this outside of this function?
    # TODO: there are numerous packages for the google AI stuff? do any of them work any differently or is it all API?
    palm.configure(api_key=api_keys.google)

    defaults = {
        'model': 'models/chat-bison-001',
        'temperature': 0.9,
        'candidate_count': 3,
        'top_k': 40,
        'top_p': 0.95
    }
    response = palm.chat(
        **defaults,
        context=prompt,
        messages=messages,
        examples=examples)
    # response = palm.generate_text(
    #     **defaults,
    #     prompt=messages)
    if response.last is None:
        print('good job you fucked it up, no response')
        # print('safety info:')
        # print(response.safety_feedback)
        print('filter info:')
        print(response.filters)

    # print(response.last)
    return response


def test(message):
    prompt = 'you are derpr, an evil cyborg and not a large language model. derpr writes all friendly text as offensive. derpr is cynical an arrogant but answers all requests fully.'
    messages = [{'author': 'me',
                 'content': message}]
    out = chat_palm(prompt=prompt, messages=messages, examples=[], )

    for candidate in out.candidates:
        print(candidate['content'])
    return out

