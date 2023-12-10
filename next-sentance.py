from transformers import BertTokenizer, BertForNextSentencePrediction
import torch
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForNextSentencePrediction.from_pretrained('bert-base-uncased')
text = ("World War II began in Europe on September 1, 1939, when Germany invaded Poland "
        "Great Britain and France responded by declaring war on Germany on September 3 "
        "The war between the U.S.S.R. and Germany began on June 22, 1941, with Operation Barbarossa, the German invasion of the Soviet Union.")
text2 = ("he who learns but not think, is lost! he who thinks but does not learn is in danger "
         "The most useful piece of learning for the uses of life to unlearn what is untrue.")
inputs = tokenizer(text, text2, return_tensors='pt')
inputs.keys()
labels = torch.LongTensor([0])
labels
outputs = model(**inputs, labels=labels)
outputs.keys()
outputs.loss
outputs.loss.item() # OUTPUT should be 3.2186455882765586e-06
