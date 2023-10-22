```python3 nlp.py source_lang_pdf target_lang_pdf```
Due to time limit, there is another more complex version that is incomplete but could make more sense in /langchain
The available version above is a simple trial based on traditional pdf reading method but is not performing well.
For translation data reading and alignment, we could use langchain+openai.
For the model side, we could first try finetuning the model. If the result is not as good as our expectation, we could try keeping only the first few layers or freeze the pretrained parameters and add our own model structure to finetune.