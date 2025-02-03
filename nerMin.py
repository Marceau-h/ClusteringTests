import json

import spacy

from errors import BigBoi

# nlp_sm = spacy.load('fr_core_news_sm')
# nlp_lg = spacy.load('fr_core_news_lg')
# nlp_por_sm = spacy.load('pt_core_news_sm')
# nlp_por_lg = spacy.load('pt_core_news_lg')
# nlp_en_sm = spacy.load('en_core_web_sm')
# nlp_en = spacy.load('en_core_web_lg')

nlp_models = {
    'fr': {
        'sm': "fr_core_news_sm",
        'lg': "fr_core_news_lg"
    },
    'pt': {
        'sm': "pt_core_news_sm",
        'lg': "pt_core_news_lg"
    },
    'en': {
        'sm': "en_core_web_sm",
        'lg': "en_core_web_lg"
    }
}

DEFAULT_NLP_MAX_LENGTH = 1_000_000
TRESHOLD_NLP_MAX_LENGTH = 5_000_000_000

def nerMin(
        text:str,
        model:str='sm',
        lang:str='fr',
        enforce_nlp_length:bool=False,
        nlp_max_length:int=DEFAULT_NLP_MAX_LENGTH,
        treshold:int=TRESHOLD_NLP_MAX_LENGTH,
):
    """
    Extract named entities from a text
    :param text: The text to analyze
    :param model: The model to use (sm or lg)
    :param lang: The language of the text (fr, pt or en)
    :param enforce_nlp_length: Whether to enforce the NLP length limit (recursive call doubling the max_length each time)
    :return: The named entities found
    """
    # if lang == 'fr':
    #     if model == 'sm':
    #         nlp = nlp_sm
    #     else:
    #         nlp = nlp_lg
    # elif lang == 'pt':
    #     if model == 'sm':
    #         nlp = nlp_por_sm
    #     else:
    #         nlp = nlp_por_lg
    # else:
    #     if model == 'sm':
    #         nlp = nlp_en_sm
    #     else:
    #         nlp = nlp_en
    nlp = spacy.load(nlp_models[lang][model])

    if DEFAULT_NLP_MAX_LENGTH != nlp_max_length:
        if nlp_max_length > treshold:
            raise BigBoi(f"Text is too long: {nlp_max_length} > {treshold}")
        nlp.max_length = nlp_max_length

    try:
        doc = nlp(text)
        return {
            ent.text.strip()
            for ent in doc.ents
            if ent.label_ in {
                'LOC',
                'GPE',
            }
        }
    except ValueError as e:
        if enforce_nlp_length:
            print(f"Retrying with max_length = {nlp_max_length * 2:_} for {text[:100]}")
            return nerMin(text, model, lang, True, nlp_max_length * 2)
        else:
            raise BigBoi(f"Text is too long: {e}")


if __name__ == "__main__":
    text = """
    Bonjour, je m'appelle Jean Dupont et j'habite à Paris. 
    Je suis né le 1er janvier 1980.
    Je travaille chez Google depuis 2010.
    J'ai un chat et un chien.
    Qui s'appellent respectivement Parisle et Médor.

    As-tu vu ? As-tu vu les quenouilles ? WoaaW, as-tu vu les belles quenouilles ? Hein ? Les belles quenouilles dans le marécage, spectaculaire, renversant ! As-tu vu ? Les voilà, les voilà. Hein ? Quelles belles quenouilles.
    Dans ces quenouilles, il y a des grenouilles, des crapauds. Il y a le sexe, il y a les mini-grenouilles, les mini-crapauds.. hey ! As-tu vu ?!
    Voilà, voilà ! Voi.. as-tu vu les quenouilles ? As-tu vu les quenouilles, woaaaw ! Hey, as-tu vu cette quenouille ? Cette quenouille vient d'un monde extérieur, planète B, planète C, je ne le sais pas. MAIS, qu'est-ce que je sais par exemple, c'est que ces quenouilles importées d'Italie qui font face à l'Afrique du sud.. sont là !! Ils sont là !! C'est les quenouilles !! Heey !! Voilà !! Voilà les quenouilles !! As-tu vu ?! As-tu.. hey hey, as-tu vu les quenouilles ?
    Aga, position en bas, position en haut, les quenouilles à gauche, les quenouilles à droite, ohh ohh, en haut, je saute, je me couche à terre. As-tu vu ? As-tu vu les quenouilles ?!
    Oooh, hey, hey, hey, oh, oh, oh, oh, heeAAAAAAAAaaAAAH Heeaaaah heu heu ouh ouh heu HaaaaAAAAAAAaah as-tu vu les quenouilles ?! Woaaw, hey, ohohoh, voilà, voilà, voilà les quenouilles.. oheyy.. C'est spectaculaire, c'est renversant. Hey, oh, awé..
    ..as-tu vu ? As-tu.. attends attends attends, agagagaga agagagagagagaga as-tu vu ? As-tu vu les quenouilles ? Très jolies quenouilles ! Très jolies ! WoaaW oooh, ooh..
    Hey johny !
    """

    with open("sm_entities.json", "w", encoding="utf-8") as f:
        json.dump(list(nerMin(text, model='sm')), f, ensure_ascii=False, indent=4)

    with open("lg_entities.json", "w", encoding="utf-8") as f:
        json.dump(list(nerMin(text, model='lg')), f, ensure_ascii=False, indent=4)
