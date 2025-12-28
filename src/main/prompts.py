extract_relation_prompt = """Please provide as few highly relevant relations as possible to the question and its subobjectives from the following relations (separated by semicolons).
Here is an example:
Q: Name the president of the country whose main spoken language was Brahui in 1980?
Subobjectives: ['Identify the countries where the main spoken language is Brahui', 'Find the president of each country', 'Determine the president from 1980']
Topic Entity: Brahui Language
Relations: language.human_language.main_country; language.human_language.language_family; language.human_language.iso_639_3_code; base.rosetta.languoid.parent; language.human_language.writing_system; base.rosetta.languoid.languoid_class; language.human_language.countries_spoken_in; kg.object_profile.prominent_type; base.rosetta.languoid.document; base.ontologies.ontology_instance.equivalent_instances; base.rosetta.languoid.local_name; language.human_language.region
The output is: 
['language.human_language.main_country','language.human_language.countries_spoken_in','base.rosetta.languoid.parent']

Now you need to directly output relations highly related to the following question and its subobjectives in list format without other information or notes.
Q: """

system_prompt = "You are an AI assistant that helps people find information."


prune_entity_prompt = """
Which entities in the following list ([] in Triples) can be used to answer question? Please provide the minimum possible number of entities, and strictly adhering to the constraints mentioned in the question. Remember at least keep one entity!
Here is an example:
Q: The movie featured Miley Cyrus and was produced by Tobin Armbrust?
Triplets: Tobin Armbrust film.producer.film ['The Resident', 'So Undercover', 'Let Me In', 'Begin Again', 'The Quiet Ones', 'A Walk Among the Tombstones']
Output: ['So Undercover']

Now you need to directly output the entities from [] in Triplets for the following question in list format without other information or notes.
Q: """

