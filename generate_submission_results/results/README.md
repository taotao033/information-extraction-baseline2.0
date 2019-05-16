#### f1_0.814
first, you need to do entity identification by using Information_Extraction/information-extraction-baseline2.0/entities_extraction/bert-chinese-ner.
second, you need to disambiguate the entity by using Information_Extraction/information-extraction-baseline2.0/bilstm_confirm_p_in_entities2.py
the last, you need to do multiple label relationship classification by using Information_Extraction/information-extraction-baseline2.0/relationship_extraction/entity-aware-relation-classification_49_multilabels_rel