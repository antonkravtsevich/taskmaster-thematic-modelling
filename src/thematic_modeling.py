import gensim


class ThematicModeller:

    def __init__(self, dict_path, model_path):
        self.dictionary = gensim.corpora.Dictionary.load_from_text(dict_path)
        self.model = gensim.models.ldamodel.LdaModel.load(model_path)

    def get_text_thems(self, text):
        text_lower = text.lower()
        text_splitted = text_lower.split()
        bow = self.dictionary.doc2bow(text_splitted)
        themes_raw = self.model.get_document_topics(bow)
        themes = []
        for theme in themes_raw:
            themes.append({
                'theme_number': theme[0],
                'theme_conformity': float(theme[1])
            })
        return themes

    def get_theme_words(self, theme_number):
        words_raw = self.model.show_topic(theme_number)
        words = []
        for word in words_raw:
            words.append({
                'word': word[0],
                'word_importance': float(word[1])
            })
        return words
