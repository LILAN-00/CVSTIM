from GroundingDINO.groundingdino.util.inference import load_image, predict

def locate_obj(image_filename, text_prompts, model):
    BOX_TRESHOLD=0.1   
    TEXT_TRESHOLD=0.25 
    image_source, image = load_image(image_filename)
    obj_num=len(text_prompts)
    input_sentence=' . '.join(text_prompts)+' .'
    boxes, logits, phrases = predict(
        model=model,
        image=image,
        caption=input_sentence,
        box_threshold=BOX_TRESHOLD,
        text_threshold=TEXT_TRESHOLD
    )   ## boxes:cxcywh
    return boxes