from natasha import (
    Segmenter,
    MorphVocab,
    
    NewsEmbedding,
    NewsMorphTagger,
    NewsSyntaxParser,
    NewsNERTagger,
    
    PER,
    NamesExtractor,
    AddrExtractor,

    Doc
)

class NER:
    def __init__(self):
        self.__segmenter = Segmenter()
        self.__morph_vocab = MorphVocab()

        emb = NewsEmbedding()
        self.__morph_tagger = NewsMorphTagger(emb)
        self.__syntax_parser = NewsSyntaxParser(emb)
        self.__ner_tagger = NewsNERTagger(emb)

        self.__names_extractor = NamesExtractor(self.__morph_vocab)
        self.__addr_extractor = AddrExtractor(self.__morph_vocab)
    
    def get_ner(self, incident):
        doc = Doc(incident)
        doc.segment(self.__segmenter)
        doc.tag_morph(self.__morph_tagger)
        for token in doc.tokens:
            token.lemmatize(self.__morph_vocab)
        doc.parse_syntax(self.__syntax_parser)

        doc.tag_ner(self.__ner_tagger)

        for span in doc.spans:
            span.normalize(self.__morph_vocab)
    
        for span in doc.spans:
            if span.type == PER:
                span.extract_fact(self.__names_extractor)

        persons = []

        for person in doc.spans:
            if person.type == PER:
                person_dict = person.fact.as_dict
                if list(person_dict.keys()) == ['first']:
                    persons.append(person_dict['first'])
                if list(person_dict.keys()) == ['first', 'last']:
                    persons.append(person_dict['first'] + ' ' + person_dict['last'])
                if list(person_dict.keys()) == ['first', 'last', 'middle']:
                    persons.append(person_dict['first'] + ' ' + person_dict['middle'] + ' ' +person_dict['last'])
    
        addrs = []
    
        if self.__addr_extractor.find(incident) != None:
            for addr in self.__addr_extractor.find(incident).fact.parts:
                addrs.append(addr.type + ' ' + addr.value)
    
        return [persons, addrs]