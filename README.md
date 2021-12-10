# Self-attentive classification-based anomaly detection in unstructured logs

This repository is the **unofficial** implementation of [Self-attentive classification-based anomaly detection in unstructured logs](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9338283&casa_token=aXQCQgTo9YYAAAAA:NJJhyLJgWfkeAjPCkMFAJYWmXyqLUk10u4RtYzT6HYVx0M-YZyq5nUU3BwyQt-GBgyTI51WYfGc&tag=1). 

>📋  Please find a demo Colab notebook at the src folder at project root

## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

>📋  When using demo notebook, simply import the notebook along with the utils.py (data preprocessor) and modify the folder path to point to your [datasets](https://www.usenix.org/cfdr-data).

Baselines: We implemented two baselines used in the paper - PCA and Deeplog. Please refer to corresponding notebooks for their specifics.
 
## Training

To train the model(s) in the paper, run this command:

Run the notebook with TPU runtime and parallel execution strategy on, each epoch at batch size 512 will take less than 1min for first 5 million rows of data.

## Evaluation

The results can be evaluated by observing the F1-score, Recall, Precision and Accuracy. 

## Results [WIP]

Our model achieves the following performance on the three target datasets used by the original paper:

WIP

## To cite the original paper

<pre><code>@article{nedelkoski2020self,
  title={Self-Attentive Classification-Based Anomaly Detection in Unstructured Logs},
  author={Nedelkoski, Sasho and Bogatinovski, Jasmin and Acker, Alexander and Cardoso, Jorge and Kao, Odej},
  journal={arXiv preprint arXiv:2008.09340},
  year={2020}
}
</code></pre>

## To cite our work
Please click on the button `Cite this repository` below the repo description. A bibitex will be generated for your convinience.

## License and contributions
This code is released under MIT License.

Pull requests and issues are welcomed to enhance the implementation.

