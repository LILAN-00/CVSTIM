def construct_prompt(prompt, all_objs, real_objs):
    str_text=', '.join(f'{obj}' for obj in all_objs if obj not in real_objs)
    return prompt.format(text=str_text)