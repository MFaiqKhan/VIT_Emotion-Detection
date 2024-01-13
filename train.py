import argparse
from ai.dive.data.label_reader import LabelReader
import os
from ai.dive.data.image_file_classification import ImageFileClassificationDataset
from transformers import ViTForImageClassification, ViTImageProcessor, TrainingArguments, Trainer
from datasets import load_metric
import torch
import numpy as np

def main():
    # parse command line arguments
    parser = argparse.ArgumentParser(description='Train a ViT on dataset')
    parser.add_argument('-d', '--data', required=True, type=str, help='datasets to train/eval model on')
    parser.add_argument('-o', '--output', required=True, type=str, help='output file to write results to')
    parser.add_argument('-m', '--base_model', default="google/vit-base-patch16-224-in21k", type=str, help='The base model to use')
    parser.add_argument('-g', '--gpu', default=False, help='Train on the GPU if supported')
    args = parser.parse_args()
    print(torch.cuda.is_available())

    labels_file = os.path.join(args.data, "labels.txt")
    label_reader = LabelReader(labels_file)
    labels = label_reader.labels()
    print(labels)

    # Same processor as before that is tied to the model
    processor = ViTImageProcessor.from_pretrained(args.base_model)

    # Load the dataset into memory, and convert to a hugging face dataset
    print("Preparing train dataset...")
    print(os.path.join(args.data, "train.csv"))
    train_file = os.path.join(args.data, "train.csv")
    ds = ImageFileClassificationDataset(
        data_dir = args.data,
        file = train_file, 
        label_reader = label_reader, 
        img_processor = processor,
        #num_samples=100
    )
    train_dataset = ds.to_hf_dataset()

    print(train_dataset[0])
    print(train_dataset[0]['pixel_values'].shape)

    train_file = os.path.join(args.data, "test.csv")
    ds = ImageFileClassificationDataset(
        data_dir=args.data,
        file=train_file,
        label_reader=label_reader,
        img_processor=processor,
        #num_samples=100
    )
    eval_dataset = ds.to_hf_dataset()

    print(eval_dataset[0])
    print(eval_dataset[0]['pixel_values'].shape)
    print(torch.cuda.is_available())

    model = ViTForImageClassification.from_pretrained(
        args.base_model,
        num_labels=len(labels),
        id2label={str(i): c for i, c in enumerate(labels)},
        label2id={c: str(i) for i, c in enumerate(labels)}
    )

    training_args = TrainingArguments(
        output_dir=args.output, # directory to save the model
        per_device_train_batch_size=16,
        evaluation_strategy="steps",
        num_train_epochs=4, # loop through the data N times
        no_cuda=(not args.gpu), # use the GPU or not
        save_steps=1000, # save the model every N steps
        eval_steps=1000, # evaluate the model every N steps
        logging_steps=10,
        learning_rate=2e-4,
        save_total_limit=2, # only keep the last N models
        remove_unused_columns=False,
        report_to='tensorboard',
        load_best_model_at_end=True,
    )

# We must take the dataset and stack it into pytorch tensors
# Our batch size above was 16
# So this will be a stack of 16 images into a tensor
    def collate_fn(batch):
        return {
            'pixel_values': torch.stack([x['pixel_values'] for x in batch]),
            'labels': torch.tensor([x['labels'] for x in batch])
        }

    # We want to evaluate accuracy of the model on the test/eval set
    metric = load_metric("accuracy")
    def compute_metrics(p):
        return metric.compute(predictions=np.argmax(p.predictions, axis=1), references=p.label_ids)

# Instantiate a trainer with all the components we have built so far
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=collate_fn,
        compute_metrics=compute_metrics,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=processor,
    )

    # Kick off the train
    print("Training model...")
    train_results = trainer.train()
    
    trainer.save_model()
    trainer.log_metrics("train", train_results.metrics)
    trainer.save_metrics("train", train_results.metrics)
    trainer.save_state()




if __name__ == '__main__':
    main()


    """ 
    Microsoft Windows [Version 10.0.22621.2861]
(c) Microsoft Corporation. All rights reserved.

E:\Github\emotion_detection>E:\Github\emotion_detection\venv_vit_emotion\Scripts\activate

(venv_vit_emotion) E:\Github\emotion_detection>oxen df FacialEmotionRecognition/train.csv 
shape: (28_709, 2)
┌───────────────────────────────────┬─────────┐
│ file                              ┆ label   │
│ ---                               ┆ ---     │
│ str                               ┆ str     │
╞═══════════════════════════════════╪═════════╡
│ train/happy/Training_50449107.jp… ┆ happy   │
│ train/happy/Training_70433018.jp… ┆ happy   │
│ train/happy/Training_85610005.jp… ┆ happy   │
│ train/happy/Training_4460748.jpg  ┆ happy   │
│ …                                 ┆ …       │
│ train/disgust/Training_81049148.… ┆ disgust │
│ train/disgust/Training_28365203.… ┆ disgust │
│ train/disgust/Training_39197750.… ┆ disgust │
│ train/disgust/Training_12525818.… ┆ disgust │
└───────────────────────────────────┴─────────┘

(venv_vit_emotion) E:\Github\emotion_detection>oxen df FacialEmotionRecognition/train.csv --sql 'SELECT label, COUNT(*) FROM df GROUP BY label;'    
[2024-01-01T12:22:23Z ERROR liboxen::core::df::tabular] Error running sql: sql parser error: Unterminated string literal at Line: 1, Column 1
'SELECT

(venv_vit_emotion) E:\Github\emotion_detection>oxen df FacialEmotionRecognition/train.csv --sql 'SELECT label, COUNT(*) FROM df GROUP BY label;'
[2024-01-01T12:22:30Z ERROR liboxen::core::df::tabular] Error running sql: sql parser error: Unterminated string literal at Line: 1, Column 1
'SELECT

(venv_vit_emotion) E:\Github\emotion_detection>oxen df FacialEmotionRecognition/train.csv --sql "SELECT label, COUNT(*) FROM df GROUP BY label;"
shape: (7, 2)
┌──────────┬───────┐
│ label    ┆ count │
│ ---      ┆ ---   │
│ str      ┆ u32   │
╞══════════╪═══════╡
│ angry    ┆ 3995  │
│ disgust  ┆ 436   │
│ fear     ┆ 4097  │
│ happy    ┆ 7215  │
│ neutral  ┆ 4965  │
│ sad      ┆ 4830  │
│ surprise ┆ 3171  │
└──────────┴───────┘

(venv_vit_emotion) E:\Github\emotion_detection>pip install ai-dive

    [notice] A new release of pip is available: 23.1.2 -> 23.3.2
    [notice] To update, run: python.exe -m pip install --upgrade pip

(venv_vit_emotion) E:\Github\emotion_detection>python train.py -d FacialEmotionRecognition -m google/vit-base-patch16-224-in21k -o models/trained/
Loading labels...
Got 7 labels
['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
 
     """
    



    """ ----------PowerShell------------

    PS E:\Github\emotion_detection> E:\Github\emotion_detection\venv_vit_emotion\Scripts\Activate.ps1
source : The term 'source' is not recognized as the name of a cmdlet, function, script file, or operable program. Check the spelling of the name, 
or if a path was included, verify that the path is correct and try again.                                                                           At line:1 char:1                                                                                                                                    + source E:\Github\emotion_detection\venv_vit_emotion\Scripts\Activate. ...                                                                         + ~~~~~~                                                                                                                                            
    + CategoryInfo          : ObjectNotFound: (source:String) [], CommandNotFoundException
    + FullyQualifiedErrorId : CommandNotFoundException
 
PS E:\Github\emotion_detection> E:\Github\emotion_detection\venv_vit_emotion\Scripts\Activate.ps1       
(venv_vit_emotion) PS E:\Github\emotion_detection> Invoke-WebRequest -Uri "https://github.com/Oxen-AI/Oxen/releases/download/v0.9.17/oxen.exe" 
   

StatusCode        : 200
StatusDescription : OK
Content           : {77, 90, 144, 0...}
RawContent        : HTTP/1.1 200 OK
                    Connection: keep-alive
                    Content-MD5: Q3an5AWRTVj1y+KrUAObuA==
                    x-ms-request-id: 81bd94af-001e-003a-6204-3ffb2d000000
                    x-ms-version: 2020-04-08
                    x-ms-creation-time: Wed, 27 Dec 2023 20...
Headers           : {[Connection, keep-alive], [Content-MD5, Q3an5AWRTVj1y+KrUAObuA==], [x-ms-request-id, 81bd94af-001e-003a-6204-3ffb2d000000],    
                    [x-ms-version, 2020-04-08]...}
RawContentLength  : 69799424



(venv_vit_emotion) PS E:\Github\emotion_detection> oxen clone https://hub.oxen.ai/ox/FacialEmotionRecognition
oxen : The term 'oxen' is not recognized as the name of a cmdlet, function, script file, or operable program. Check the spelling of the name, or 
if a path was included, verify that the path is correct and try again.
At line:1 char:1
+ oxen clone https://hub.oxen.ai/ox/FacialEmotionRecognition
+ ~~~~
    + CategoryInfo          : ObjectNotFound: (oxen:String) [], CommandNotFoundException
    + FullyQualifiedErrorId : CommandNotFoundException

(venv_vit_emotion) PS E:\Github\emotion_detection> Invoke-WebRequest -Uri "https://github.com/Oxen-AI/Oxen/releases/download/v0.9.17/oxen.exe" -OutFile "E:\Github\emotion_detection"                                                                                                                   Invoke-WebRequest : Access to the path 'E:\Github\emotion_detection' is denied.                                                                     At line:1 char:1                                                                                                                                    + Invoke-WebRequest -Uri "https://github.com/Oxen-AI/Oxen/releases/down ...                                                                         + ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~                                                                                 + CategoryInfo          : NotSpecified: (:) [Invoke-WebRequest], UnauthorizedAccessException                                                    
    + FullyQualifiedErrorId : System.UnauthorizedAccessException,Microsoft.PowerShell.Commands.InvokeWebRequestCommand

(venv_vit_emotion) PS E:\Github\emotion_detection> Invoke-WebRequest -Uri "https://github.com/Oxen-AI/Oxen/releases/download/v0.9.17/oxen.exe" -OutFile "E:\Github\emotion_detection\oxen.exe"
(venv_vit_emotion) PS E:\Github\emotion_detection> Invoke-WebRequest -Uri "https://github.com/Oxen-AI/Oxen/releases/download/v0.9.17/oxen-server.exe" -OutFile "E:\Github\emotion_detection\oxen-server.exe"
>> 
(venv_vit_emotion) PS E:\Github\emotion_detection> oxen clone https://hub.oxen.ai/ox/FacialEmotionRecognition
oxen : The term 'oxen' is not recognized as the name of a cmdlet, function, script file, or operable program. Check the spelling of the name, or 
if a path was included, verify that the path is correct and try again.
At line:1 char:1
+ oxen clone https://hub.oxen.ai/ox/FacialEmotionRecognition
+ ~~~~
    + CategoryInfo          : ObjectNotFound: (oxen:String) [], CommandNotFoundException
    + FullyQualifiedErrorId : CommandNotFoundException


Suggestion [3,General]: The command oxen was not found, but does exist in the current location. Windows PowerShell does not load commands from the current location by default. If you trust this command, instead type: ".\oxen". See "get-help about_Command_Precedence" for more details.
(venv_vit_emotion) PS E:\Github\emotion_detection> .\oxen clone https://hub.oxen.ai/ox/FacialEmotionRecognition
Err: Directory already exists: FacialEmotionRecognition
(venv_vit_emotion) PS E:\Github\emotion_detection> .\oxen clone https://hub.oxen.ai/ox/FacialEmotionRecognition
🐂 Oxen pull origin main
FacialEmotionRecognition (59.9 MB) contains 35896 files

  📊 tabular (5)        📄 text (4)     📸 image (35887)

Fetching commits for main
🐂 Downloading 59.9 MB

🎉 cloned https://hub.oxen.ai/ox/FacialEmotionRecognition to FacialEmotionRecognition/

(venv_vit_emotion) PS E:\Github\emotion_detection> oxen --version
oxen : The term 'oxen' is not recognized as the name of a cmdlet, function, script file, or operable program. Check the spelling of the name, or 
if a path was included, verify that the path is correct and try again.
At line:1 char:1
+ oxen --version
+ ~~~~
    + CategoryInfo          : ObjectNotFound: (oxen:String) [], CommandNotFoundException
    + FullyQualifiedErrorId : CommandNotFoundException


Suggestion [3,General]: The command oxen was not found, but does exist in the current location. Windows PowerShell does not load commands from the current location by default. If you trust this command, instead type: ".\oxen". See "get-help about_Command_Precedence" for more details.
(venv_vit_emotion) PS E:\Github\emotion_detection> 


The error message `FileNotFoundError: [Errno 2] No such file or directory: 'FacialEmotionRecognition\\FacialEmotionRecognition\\train.csv'` is indicating that Python is 
unable to find the file `train.csv` in the directory `'FacialEmotionRecognition\\FacialEmotionRecognition'`.
The double backslashes in the path (`\\`) are used because in Python, a single backslash is an escape character. 
This means it is used to introduce special character sequences. For example, `\n` is a newline, and `\t` is a tab. 
So, when Python sees the `\F` in your path, it thinks you're trying to start a special character sequence, 
which causes an error. By using double backslashes (`\\`), you're telling Python to treat the subsequent character as a normal character, 
not as the start of an escape sequence [Source 2](https://discuss.python.org/t/filenotfounderror-errno-2-no-such-file-or-directory/3549).
So, in your case, the path `'FacialEmotionRecognition\\FacialEmotionRecognition\\train.csv'` is actually 
referring to a file named `train.csv` located in a directory named `FacialEmotionRecognition\FacialEmotionRecognition`. 
If the file is not located in this directory, you will get a `FileNotFoundError` [Source 4](https://www.pythoncheatsheet.org/cheatsheet/file-directory-path).
From your code, it seems like you're passing `FacialEmotionRecognition` as the `-d` argument when running your script. 
This means that Python is looking for the `train.csv` file in a directory named `FacialEmotionRecognition` that is located in the current working directory.
 However, based on the information you've provided, the `train.csv` file is located in a directory named `FacialEmotionRecognition` that 
 is a sibling of the directory containing your `train.py` script.
To fix this issue, you should modify the `-d` argument when running your script to include the full path to the `FacialEmotionRecognition` directory. For example:

```bash
(venv_vit_emotion) E:\Github\emotion_detection>python train.py -d E:\Github\emotion_detection\FacialEmotionRecognition -m google/vit-base-patch16-224-in21k -o models/trained/
```

In this command, `-d E:\Github\emotion_detection\FacialEmotionRecognition` tells the script to look for the `train.csv` file 
in the `E:\Github\emotion_detection\FacialEmotionRecognition` directory [Source 1](https://docs.python.org/3/library/argparse.html), 
[Source 4](https://docs.python.org/3/howto/argparse.html).
    """



