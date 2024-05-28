# Train transformer models for inferring article text embedding and computing pairwise similarity

1. **truncate_text.py** is to truncate the news article to fit its input length into transformer models.

2. **weight_link_inference.py** (weight_link_inference_script.sh) is to load pretrained models and fine-tune them on annotated data. 

3. **model_evaluation.py** (model_evaluation_script.sh) is to evaluate performance of each model, so that you can compare them and find the best one in the end.

4. **load_model_inference.py** (load_model_inference_script.sh) is to load the best fine-tuned model and apply it to the entire data. So that we can form a large network between the news articles/social polls.

5. **graph clustering** -- we can further apply OSLOM clustering algorithm for the large network to get event/topic clusters that are much more interpretable than numeric weighted links, and it will also mitigate the dropped link in our primary filters for effeciently processing the large dataset. The **both reasons are vital** for the entire global graph analysis.
