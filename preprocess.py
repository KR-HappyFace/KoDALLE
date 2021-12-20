# -*- coding:utf8 -*-

def remove_style(input_text: str) -> str:
    # split sentences by .
    sentences = input_text.split(".")
    return_sentences = []

    for sentence in sentences:
        if "스타일" in sentence:
            pass
        else:
            return_sentences.append(sentence)
    # join sentences into one str
    return ".".join(return_sentences).strip()

def remove_subj(input_text: str) -> str:
    result = []
    subj = []
    text_data = input_text.split(' ')
    for d in text_data:
      if "에서" in d:
        if d in subj: continue
        subj.append(d)
      result.append(d)

    return ' '.join(result)

