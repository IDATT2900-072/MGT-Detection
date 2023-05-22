from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

output_path = "andreas122001/roberta-academic-detector"
tokenizer = AutoTokenizer.from_pretrained(output_path)
model = AutoModelForSequenceClassification.from_pretrained(output_path)

classifier = pipeline("text-classification", 
                      model=model, 
                      tokenizer=tokenizer
                      )

text_in = ""
while text_in != 'q':
    text_in = input("Text to classify (q to exit): ")
    print(classifier(text_in))
