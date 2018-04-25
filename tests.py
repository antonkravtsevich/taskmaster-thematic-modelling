from src.thematic_modeling import ThematicModeller
import unittest


class TestSentimentProcessing(unittest.TestCase):

    def test_load(self):
        tm = ThematicModeller(
            dict_path='./src/data/wiki_en_output_wordids.txt.bz2',
            model_path='./src/data/wiki_lda.pkl')
        self.assertIsNotNone(tm)

    def test_words_of_regular_theme(self):
        tm = ThematicModeller(
            dict_path='./src/data/wiki_en_output_wordids.txt.bz2',
            model_path='./src/data/wiki_lda.pkl')
        words_info = tm.get_theme_words(36)
        words = []
        for pair in words_info:
            words.append(pair['word'])
        self.assertIn('game', words)
        self.assertIn('gameplay', words)
        self.assertIn('player', words)
        self.assertIn('video', words)

    def test_theme_modelling(self):
        tm = ThematicModeller(
            dict_path='./src/data/wiki_en_output_wordids.txt.bz2',
            model_path='./src/data/wiki_lda.pkl')

        text = 'A video game is an electronic game that involves '\
            'interaction with a user interface to generate visual '\
            'feedback on a video device such as a TV screen or computer '\
            'monitor. The word video in video game traditionally '\
            'referred to a raster display device, but as of the 2000s, '\
            'it implies any type of display device that can produce '\
            'two- or three-dimensional images. Some theorists categorize '\
            'video games as an art form, but this designation is '\
            'controversial.'
        themes_info = tm.get_text_thems(text)
        themes = []
        for pair in themes_info:
            themes.append(pair['theme_number'])
        self.assertIn(31, themes)
        self.assertIn(36, themes)
        self.assertIn(55, themes)


if __name__ == '__main__':
    unittest.main()
