# Text Summarizer Project

This project demonstrates how to use pre-trained models from Hugging Face's `transformers` library to perform text summarization using the **SAMSum** dataset, which consists of conversational dialogues. It leverages state-of-the-art NLP models to create concise summaries from dialogues.

## Project Structure

The main components of the project include:

### 1. **Environment Setup**
   The project installs necessary libraries from Hugging Face and other key libraries, such as:
   - `transformers`: For accessing pre-trained models.
   - `datasets`: For loading and handling datasets.
   - `sacrebleu`: For evaluating text summarization.
   - `py7zr`: For handling dataset compression.
   - `evaluate`: A new library from Hugging Face for evaluation, replacing `rouge_score`.
   - `accelerate`: For optimizing model performance on GPUs.

   Example setup commands:
   ```python
   !pip install transformers[sentencepiece] datasets sacrebleu evaluate py7zr -q
   !pip install --upgrade accelerate
   ```

### 2. **Hugging Face Model Integration**
   The project uses Hugging Face's `AutoModelForSeq2SeqLM` and `AutoTokenizer` to load pre-trained summarization models.

   Hugging Face is used for its ease of access to a wide variety of pre-trained models, making it simpler to implement NLP tasks like summarization. The library supports models like BART, T5, and Pegasus, which are widely used for text summarization.

### 3. **Data Loading**
   The project utilizes the `datasets` library to load the **SAMSum** dataset. The SAMSum dataset contains human-written summaries of dialogues, making it ideal for testing conversational summarization models.

   Example:
   ```python
   from datasets import load_dataset
   dataset = load_dataset("samsum")
   ```

### 4. **Tokenization and Summarization**
   Text inputs (from dialogues) are tokenized into model-compatible formats using the loaded tokenizer. After tokenization, the summarization model processes the input and generates summaries.

   Example:
   ```python
   inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True)
   summary_ids = model.generate(inputs.input_ids)
   summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
   ```

### 5. **Evaluation**
   The project evaluates the generated summaries using metrics like ROUGE, which is now handled by Hugging Face's `evaluate` library. This library simplifies evaluation and supports various metrics.

   Example:
   ```python
   from evaluate import load
   rouge = load("rouge")
   ```

## Why Hugging Face is Used
Hugging Face provides pre-trained models and tools that significantly reduce the complexity of working with deep learning models for NLP tasks. By using `transformers`, the project gains access to:
- Pre-trained state-of-the-art models.
- Tools for easily fine-tuning these models on custom datasets.
- Seamless integration with GPU acceleration using `accelerate`.
- Access to an extensive collection of NLP datasets through the `datasets` library.

## Running the Project in Google Colab

To run this project in Google Colab, follow these steps:

1. **Open Google Colab**: Go to [Google Colab](https://colab.research.google.com/).

2. **Upload the Notebook**: Click on `File > Upload Notebook` and upload the `Text_Summarizer_project.ipynb` file.

3. **Install Dependencies**: Run the following commands in the first cell to install the necessary dependencies:
   ```python
   !pip install transformers[sentencepiece] datasets sacrebleu evaluate py7zr -q
   !pip install --upgrade accelerate
   ```

4. **Enable GPU**: 
   - Click on `Runtime > Change runtime type`.
   - Under "Hardware accelerator," select `GPU`. 
   This ensures that the model runs faster on Google's free GPU resources.

5. **Run the Notebook**: Execute the notebook cells in sequence. The summarization model will load, process the SAMSum dataset, and generate summaries. You can modify the text input to summarize custom dialogues.

## Requirements

- Python 3.8+
- `transformers` library
- `datasets` library
- `sacrebleu`, `evaluate` for evaluation
- GPU with CUDA support (optional for faster performance)

For Google Colab, no local installation is required, but if running locally, use the following requirements file:

## `requirements.txt`
```
transformers==4.24.0
datasets==2.10.0
sacrebleu==2.3.1
evaluate==0.4.0
py7zr==0.20.2
accelerate==0.15.0
torch==2.0.1
```

## How to Run Locally

1. Clone this repository.
2. Install the required packages using the `requirements.txt`:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the Jupyter Notebook to execute the code cells and generate summaries.

---

