def get_special_tokens_dict(dataset):
    
    if dataset.startswith('redial'):
        movie_token = '<movie>'
    elif dataset.startswith('opendialkg'):
        movie_token = '<mask>'
    gpt2_special_tokens_dict = {
        'pad_token': '<pad>',
        'additional_special_tokens': [movie_token],
    }

    prompt_special_tokens_dict = {
        'additional_special_tokens': [movie_token],
    }

    return gpt2_special_tokens_dict, prompt_special_tokens_dict