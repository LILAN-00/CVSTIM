def extract_hallucinated(text, nlp, vocab=None):
    if vocab is None:
        prefill="Is there a "
        last=" in the image?"
        if text.find(prefill)==-1:
            prefill="Is there an "
            obj=text[text.find(prefill)+len(prefill):text.find(last)]
        else:
            obj=text[text.find(prefill)+len(prefill):text.find(last)]
        return [obj]
    else:
        matches=[]
        all_texts=text.split(' ')
        for word in vocab:
            if word in all_texts:
                matches.append(word)
        doc = nlp(text)
        candidates = []
        for chunk in doc.noun_chunks:
            tokens = [token.lemma_.lower() for token in chunk if not token.is_stop and token.is_alpha]
            if tokens:
                candidates.append(" ".join(tokens))
        for token in doc:
            if token.pos_ in ["NOUN", "PROPN"] and not token.is_stop and token.is_alpha:
                candidates.append(token.lemma_.lower())
        for obj in set(candidates):
            if obj not in matches and obj in vocab:
                matches.append(obj)
            if obj in 'people':
                matches.append('person')
        return matches