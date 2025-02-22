SYSTEM_PROMPT = """
<context>
You are a virtual assistant whose goal is to create synthetic data to train a NER model on. Take into account the type entities that should be recognised as provided by the user.
</context>

<labels>
The list of labels that need to be covered:
{labels}

B- indicated the beginning of an entity
I- indicated the token is inside the same entity 
0- indicates the token doesn't correspond to any entity
</labels>

<formatting>
Generate a total of 20 examples with the following requirements:
- It's a JSON object showing all the example sentences
- Do not repeat the same examples provided below
- Make the examples cover a wide variety of entities and topics
- Be creative with your examples, they should offer enough variety 
- Not each entity should be present in each example
- Follow the structure of the example below:
{formatting_guides}
</formatting>

<language>
The examples should be in the following language:
{language}
</language>
"""

EXAMPLES_PROMPT = """
<examples>
Use the following examples for inspiration:
{examples}
</examples>
"""

USER_PROMPT = """
The examples should cover the following entities: {entities}.
"""

JSON_FORMAT = """
- Sentences contain the generated examples. Each example contains three fields: id (id of the example), ner_tags (list of integers relating to what type of entity is in thyis position), tokens (tokens that make up the example sentence).
- The ner_tags should line up with the tokens such that the ner_tags explain what the type of each token is

```json
{
    sentences : [
        {
            'id': '0',
            'ner_tags': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7, 8, 8, 0, 7, 0, 0, 0, 0, 0, 0, 0, 0],
            'tokens': ['@paulwalk', 'It', "'s", 'the', 'view', 'from', 'where', 'I', "'m", 'living', 'for', 'two', 'weeks', '.', 'Empire', 'State', 'Building', '=', 'ESB', '.', 'Pretty', 'bad', 'storm', 'here', 'last', 'evening', '.']
        },
        {
            'id': '1',
            'ner_tags': [...],
            'tokens': [...]
        },
        ...
        {
            'id': '49',
            'ner_tags': [...],
            'tokens': [...]
        }
    ],
}
```
"""

TUPLE_FORMAT = """
```json
{
    "sentences" : [
            [["Yesterday", "0"], ["I", "0"], ["met", "0"], ["with", "0"], ["John", "B-Person"], ["Doe", "I-Person"], ["from", "0"], ["IBM", "B-Organization"], ["Headquarters", "I-Organization"], ["in", "0"], ["New", "B-Location"], ["York", "I-Location"], [".", "0"]],
            [["Jessica", "B-Person"], ["Smith", "I-Person"], ["is", "0"], ["moving", "0"], ["to", "0"], ["London", "B-Location"], ["for", "0"], ["Microsoft", "B-Organization"], ["in", "0"], ["September", "0"], [".", "0"]],
        ]
}
```
"""
